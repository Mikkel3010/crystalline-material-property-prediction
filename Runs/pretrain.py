import json
import shutil
import time
import yaml
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Dataset, Data
from torch_geometric.nn import GATConv, global_add_pool

from barlow import BarlowTwinsLoss
from graphlist import HDFGraphList
from dataloaders_pretrain import get_train_val_test_loader, custom_collate_fn

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

random_seed = 123
torch.manual_seed(random_seed)
np.random.seed(random_seed)


class AtomInitializer(object):
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer (AtomInitializer):
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class HDFCrystalGraphDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset_path = config["dataset_path"]
        with h5py.File(self.dataset_path, 'r') as f:
            preprocessed_crystals = HDFGraphList(f)
            self.num_graphs = len(preprocessed_crystals)
        print("amount of graphs", self.num_graphs)
        self.atom_featurizer = AtomCustomJSONInitializer(
            "Data_external/atom_init_fromgithub.json")
        self.distance_expander = GaussianDistance(dmin=0, dmax=8, step=0.2)

    def len(self):
        return self.num_graphs

    def get(self, idx):
        with h5py.File(self.dataset_path, 'r') as f:
            preprocessed_crystals = HDFGraphList(f)
            graph = preprocessed_crystals[idx]

        atomic_numbers = graph.node_attributes['atomic_number']
        atom_feature_nodes_atomic_number = torch.tensor(np.vstack([
            self.atom_featurizer.get_atom_fea(atomic_numbers[i])
            for i in range(len(atomic_numbers))
        ]), dtype=torch.float)

        edge_index = torch.tensor(
            graph.edge_indices, dtype=torch.long).t().contiguous()

        edge_features = torch.tensor(np.vstack([
            self.distance_expander.expand(dist)
            for dist in graph.edge_attributes["distance"]
        ]), dtype=torch.float)

        aug_1_atom_feature_combined = atom_feature_nodes_atomic_number.detach().clone()
        aug_2_atom_feature_combined = atom_feature_nodes_atomic_number.detach().clone()

        num_nodes = atom_feature_nodes_atomic_number.shape[0]
        mask_ratio = 0.1

        mask_indices_aug_1_nodes = torch.randperm(
            num_nodes)[:int(np.ceil(num_nodes * mask_ratio))]
        mask_indices_aug_2_nodes = torch.randperm(
            num_nodes)[:int(np.ceil(num_nodes * mask_ratio))]

        aug_1_atom_feature_combined[mask_indices_aug_1_nodes] = 0
        aug_2_atom_feature_combined[mask_indices_aug_2_nodes] = 0

        num_nodes = int(graph.num_nodes.item())

        if "multiplicity" in graph.node_attributes:
            multipli = torch.tensor(
                graph.node_attributes["multiplicity"]).unsqueeze(1)
        else:
            multipli = torch.ones((num_nodes, 1))

        aug_1_atom_feature_combined = torch.cat(
            [aug_1_atom_feature_combined, multipli], dim=1)
        aug_2_atom_feature_combined = torch.cat(
            [aug_2_atom_feature_combined, multipli], dim=1)

        aug_1_edge_features = edge_features.detach().clone()
        aug_2_edge_features = edge_features.detach().clone()

        num_edges = edge_features.shape[0]

        mask_indices_aug_1_edges = torch.randperm(
            num_edges)[:int(np.ceil(num_edges * mask_ratio))]
        mask_indices_aug_2_edges = torch.randperm(
            num_edges)[:int(np.ceil(num_edges * mask_ratio))]

        aug_1_edge_features[mask_indices_aug_1_edges] = 0
        aug_2_edge_features[mask_indices_aug_2_edges] = 0

        data1 = Data(x=aug_1_atom_feature_combined,
                     edge_index=edge_index, edge_attr=aug_1_edge_features)
        data2 = Data(x=aug_2_atom_feature_combined,
                     edge_index=edge_index, edge_attr=aug_2_edge_features)

        return data1, data2, idx


class GAT_model(torch.nn.Module):
    def __init__(self, model_params):
        super().__init__()

        node_dim = model_params['node_dim']
        node_expand_dim = model_params['node_expand_dim']
        edge_expand_dim = model_params['edge_expand_dim']
        MLP1_dim = model_params['MLP1_dim']
        out_dim = model_params['MLP2_dim']
        edge_dim = model_params['edge_dim']
        dimensionslist = model_params['dimensionslist']

        self.expand_dim = nn.Linear(node_dim, node_expand_dim)
        self.expand_edge_dim = nn.Linear(edge_dim, edge_expand_dim)

        self.layer1 = GATConv(
            node_expand_dim, dimensionslist[0][1], edge_dim=edge_expand_dim, heads=3, concat=False)
        self.layer2 = GATConv(
            dimensionslist[0][1], dimensionslist[1][1], edge_dim=edge_expand_dim, heads=3, concat=False)
        self.layer3 = GATConv(
            dimensionslist[1][1], dimensionslist[2][1], edge_dim=edge_expand_dim, heads=3, concat=False)

        self.linear1 = nn.Linear(
            dimensionslist[2][1], MLP1_dim)
        self.linear2 = nn.Linear(MLP1_dim, out_dim)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        multipli = x[:, -1].view(-1, 1)
        x = x[:, :-1]

        x = self.expand_dim(x)
        edge_attr = self.expand_edge_dim(edge_attr)

        x = F.relu(x)
        edge_attr = F.relu(edge_attr)

        x = self.layer1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.layer2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.layer3(x, edge_index, edge_attr)
        x = F.relu(x)

        x = x * multipli  # scale by multiplicity

        x = global_add_pool(x, data.batch)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        return x


def _save_config_file(model_checkpoints_folder, config_file_path):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_file_path}")
    shutil.copy(config_file_path, os.path.join(
        model_checkpoints_folder, 'config.yaml'))


class Pretrain_run(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        current_time = time.strftime(
            "%Y-%m-%d_%H-%M-%S", time.localtime())
        dir_name = current_time
        dir_name = f"{self.config['preprocessor_type']}_{dir_name}"
        log_dir = os.path.join('pretrain_runs', config["task_name"], dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.criterion = BarlowTwinsLoss(self.device, self.config)

        self.dataset = HDFCrystalGraphDataset(self.config['dataset'])
        collate_fn = custom_collate_fn
        self.train_loader, self.valid_loader = get_train_val_test_loader(
            dataset=self.dataset,
            pin_memory=self.device,
            collate_fn=collate_fn,
            batch_size=self.config['batch_size']
        )

    def _get_device(self):  # copied from crystal twin

        if torch.cuda.is_available():
            print("GPU available")
        else:
            print("GPU not available")
        device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)

        return device

    def train(self):

        model = GAT_model(self.config['model_params']).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(
        ), lr=self.config['lr'], weight_decay=self.config['weight_decay'])

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.config["stepsize"], gamma=self.config["lr_gamma"])

        model_checkpoint_folder = os.path.join(
            self.writer.log_dir, 'model_checkpoints')

        _save_config_file(model_checkpoint_folder,
                          self.config['config_file_path'])

        iterations_training = 0
        validation_iterations = 0
        best_val_loss = np.inf

        for epoch in range(self.config['epochs']):
            for batch_idx, (graph_1, graph_2, _) in enumerate(self.train_loader):
                model.train()

                graph_1 = graph_1.to(self.device)
                graph_2 = graph_2.to(self.device)

                zis = model(graph_1)
                zjs = model(graph_2)
                zis = F.normalize(zis, dim=1)
                zjs = F.normalize(zjs, dim=1)

                loss = self.criterion(zis, zjs)

                if iterations_training % self.config['log_interval'] == 0:
                    if epoch % 2 == 0:
                        print(f"Train Epoch: {epoch} [{batch_idx * graph_1.num_graphs}/{len(self.train_loader.dataset)}]"
                              f"Loss: {loss.item():.6f}")
                    self.writer.add_scalar(
                        'train/loss', loss.item(), iterations_training)
                    self.writer.add_scalar(
                        'train/lr', optimizer.param_groups[0]['lr'], iterations_training)
                    self.writer.add_scalar(
                        'train/epoch', epoch, iterations_training)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                iterations_training += 1

            if iterations_training % self.config['validation_interval'] == 0:
                total_val_loss = 0.0
                total_count = 0.0
                with torch.no_grad():

                    model.eval()
                    for val_batch_idx, (data_val_aug_1, data_val_aug_2, _) in enumerate(self.valid_loader):
                        data_val_aug_1 = data_val_aug_1.to(self.device)
                        data_val_aug_2 = data_val_aug_2.to(self.device)

                        val_zis = model(data_val_aug_1)
                        val_zjs = model(data_val_aug_2)
                        val_zis = F.normalize(val_zis, dim=1)
                        val_zjs = F.normalize(val_zjs, dim=1)

                        val_loss = self.criterion(val_zis, val_zjs)

                        if validation_iterations % self.config['val_log_interval'] == 0:
                            print(f"Validation Epoch: {epoch} [{val_batch_idx * len(data_val_aug_1)}/{len(self.valid_loader.dataset)}]"
                                  f"Loss: {val_loss.item():.6f}")
                            self.writer.add_scalar(
                                'validation/loss', val_loss.item(), validation_iterations)
                            self.writer.add_scalar(
                                'validation/loss_total', val_loss.item()*data_val_aug_1.num_graphs, validation_iterations)
                            self.writer.add_scalar(
                                'validation/lr', optimizer.param_groups[0]['lr'], validation_iterations)
                            self.writer.add_scalar(
                                'validation/epoch', epoch, validation_iterations)

                        total_val_loss += val_loss.item() * data_val_aug_1.num_graphs
                        print(f"total_val_loss: {total_val_loss}")
                        total_count += data_val_aug_1.num_graphs
                    validation_iterations += 1
                    model.train()
                print(f"Validation loss: {total_val_loss:.6f}")
                print()
                if total_val_loss < best_val_loss:
                    best_val_loss = total_val_loss
                    torch.save(model.state_dict(), os.path.join(
                        model_checkpoint_folder, f'best_model_epoch_{epoch}.pth'))
                    print(f"Best model saved with loss: {best_val_loss:.6f}")

                    self.writer.add_scalar(
                        'validation/best_loss', best_val_loss, validation_iterations)
                    self.writer.add_scalar(
                        'validation/best_epoch', epoch, validation_iterations)
                    self.writer.add_scalar(
                        'validation/best_lr', optimizer.param_groups[0]['lr'], validation_iterations)

            if epoch >= 5:
                scheduler.step()
        torch.save(model.state_dict(), os.path.join(
            model_checkpoint_folder, 'last_model.pth'))


if __name__ == "__main__":
    config_path = input("Enter the path to your config.yaml file: ").strip()

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["config_file_path"] = config_path
    print(config)

    contrastive_run = Pretrain_run(config)
    contrastive_run.train()

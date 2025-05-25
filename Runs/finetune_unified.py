from matbench.bench import MatbenchBenchmark
from dataloaders_finetune import get_train_val_test_loader, custom_collate_fn
from pretrain import GAT_model
from graphlist import HDFGraphList
from torch_geometric.nn import GATConv, global_add_pool
from torch_geometric.data import Dataset, Data
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import h5py
import yaml
import time
import shutil
import json
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
        self.dataset_path = self.config["dataset_path"]
        with h5py.File(self.dataset_path, 'r') as f:
            preprocessed_crystals = HDFGraphList(f)
            self.num_graphs = len(preprocessed_crystals)
        print("amount of graphs", self.num_graphs)
        self.atom_featurizer = AtomCustomJSONInitializer(
            "Data_external/atom_init_fromgithub.json")
        self.distance_expander = GaussianDistance(dmin=0, dmax=8, step=0.2)
        dataset_name_for_csv = config["task_name"]

        self.data = pd.read_csv(
            f"Data_external/Matbench_target_data/{dataset_name_for_csv}.csv")
        print(self.data.head())
        self.index_to_rownum = {
            row["dataset_id"]: i for i, row in self.data.iterrows()
        }
        self.size = self.data.shape[0]
        print("amount of data", self.size)

    def get_train_targets_for_normalization(self, train_indexes):
        return torch.tensor(self.data.iloc[train_indexes, 2].values)

    def len(self):
        return self.size

    def get(self, idx):
        with h5py.File(self.dataset_path, 'r') as f:
            preprocessed_crystals = HDFGraphList(f)
            graph = preprocessed_crystals[idx]
        try:

            graph_id = graph.graph_attributes['dataset_id'][0].decode()
        except AttributeError:
            graph_id = graph.graph_attributes['dataset_id']
        graph_id = graph.graph_attributes['dataset_id'][0].decode()
        row_index = self.index_to_rownum[graph_id]
        dataset_id_at_row = self.data.loc[row_index, "dataset_id"]

        if dataset_id_at_row != graph_id:
            print(
                f"Mismatch: graph_id = {graph_id}, but dataset_id at row {row_index} = {dataset_id_at_row}")
        row_index = self.index_to_rownum[graph_id]

        target = self.data.iloc[row_index, 2]
        target = torch.tensor(target, dtype=torch.float32)

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

        num_nodes = int(graph.num_nodes.item())

        if "multiplicity" in graph.node_attributes:
            multipli = torch.tensor(
                graph.node_attributes["multiplicity"]).unsqueeze(1)
        else:
            multipli = torch.ones((num_nodes, 1))

        atom_feature_nodes_atomic_number = torch.cat(
            [atom_feature_nodes_atomic_number, multipli], dim=1
        )

        atom_feature_nodes_atomic_number = atom_feature_nodes_atomic_number.float()

        data1 = Data(x=atom_feature_nodes_atomic_number,
                     edge_index=edge_index, edge_attr=edge_features)

        return data1, target, idx


class Gat_with_extra_head(nn.Module):
    def __init__(self, gat_params, classification):
        super(Gat_with_extra_head, self).__init__()
        self.classification = classification
        input_final_layer = gat_params["MLP2_dim"]
        self.GAT_model = GAT_model(gat_params)

        self.fine_tune_linear = nn.Linear(
            input_final_layer, input_final_layer)

        if classification == "classification":
            self.final_linear_classification = nn.Linear(
                input_final_layer, gat_params['fine_tune_classes'])
        elif classification == "regression":
            self.final_linear_regression = nn.Linear(
                gat_params['fine_tune_classes'], 1)

    def forward(self, data):
        x = self.GAT_model(data)
        x = F.relu(x)
        x = self.fine_tune_linear(x)
        if self.classification == "classification":
            x = self.final_linear_classification(x)
            x = F.softmax(x, dim=1)
        elif self.classification == "regression":
            x = self.final_linear_regression(x)

        return x


def _save_config_file(model_checkpoints_folder, config_file_path):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_file_path}")
    shutil.copy(config_file_path, os.path.join(
        model_checkpoints_folder, 'config.yaml'))


class Normalizer(object):  # from crystal twin paper
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean


class Finetune(object):
    def __init__(self, config):

        self.config = config
        self.device = self._get_device()

        current_time = time.strftime(
            "%Y-%m-%d_%H-%M-%S", time.localtime())

        dir_name = f"{self.config['preprocessor_type']}_{current_time}"
        base_dir = 'shuffle_Finetune_runs'

        if self.config["do_testing"]:
            base_dir = os.path.join(base_dir, "tests")

        log_dir = os.path.join(base_dir, config["task_name"], dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"working on task: ", config['task_name'])
        if config["task_type"] == "classification":
            print("classification task")
            self.criterion = nn.CrossEntropyLoss()
        elif config["task_type"] == "regression":
            print("regression task")
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(
                "Invalid task type. Must be 'classification' or 'regression'.")

        self.dataset = HDFCrystalGraphDataset(self.config['dataset'])
        # [8989, 4450, 13752, 12082, 18849]
        collate_fn = custom_collate_fn
        self.train_loader, self.valid_loader, self.test_loader, train_indexes_for_target_norm = get_train_val_test_loader(
            dataset=self.dataset,
            pin_memory=self.device,
            collate_fn=collate_fn,
            batch_size=self.config['batch_size'],
            train_val_test_ratio=config['train_val_test_ratio'],
        )
        print("first 5 train indexes: ", train_indexes_for_target_norm[:5])

        self.normalizer = Normalizer(
            self.dataset.get_train_targets_for_normalization(train_indexes_for_target_norm))

        self.config["normalizer_vales"] = {
            "mean": self.normalizer.mean.item(),
            "std": self.normalizer.std.item()}

    def _get_device(self):  # copied from crystal twin

        if torch.cuda.is_available():
            print("GPU available")
        else:
            print("GPU not available")
        device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)

        return device

    def train(self):

        model = Gat_with_extra_head(
            self.config['model_params'],
            self.config["task_type"]).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay'])

        state_dict = torch.load(self.config["saved_model_path"])
        if not self.config["do_testing"]:
            prefixed_state_dict = {
                f"GAT_model.{k}": v for k, v in state_dict.items()}
        else:
            prefixed_state_dict = state_dict

        incompatible_keys = model.load_state_dict(
            prefixed_state_dict, strict=False)
        print("Model loaded.")

        for key in incompatible_keys.missing_keys:
            print(f"Missing key: {key}")
        for key in incompatible_keys.unexpected_keys:
            print(f"Unexpected key: {key}")

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config["stepsize"],
            gamma=self.config['lr_gamma'])

        model_checkpoint_folder = os.path.join(
            self.writer.log_dir, 'model_checkpoints')
        _save_config_file(model_checkpoint_folder,
                          self.config['config_file_path'])

        iterations_training = 0
        validation_iterations = 0
        best_val_loss = np.inf
        best_val_mae = np.inf
        best_epoch_for_val = 0

        start_training_time = time.time()

        if self.config["do_training"]:
            for epoch in range(self.config['epochs']):
                model.train()

                epoch_mae_sum = 0.0
                epoch_mse_sum = 0.0
                epoch_sample_count = 0

                for batch_idx, (train_graph_1, train_targets, _) in enumerate(self.train_loader):
                    train_graph_1 = train_graph_1.to(self.device)

                    if self.config["task_type"] == "regression":
                        train_norm_targets = self.normalizer.norm(
                            train_targets).to(self.device)
                    elif self.config["task_type"] == "classification":
                        train_norm_targets = train_targets.to(self.device)

                    train_output = model(train_graph_1).squeeze(-1)
                    train_loss = self.criterion(
                        train_output, train_norm_targets)

                    with torch.no_grad():
                        if self.config["task_type"] == "regression":
                            train_output_denorm = self.normalizer.denorm(
                                train_output)
                            train_denorm_targets = self.normalizer.denorm(
                                train_norm_targets)

                            train_abs_errors = torch.abs(
                                train_output_denorm - train_denorm_targets)
                            train_squared_errors = (
                                train_output_denorm - train_denorm_targets) ** 2

                            epoch_mae_sum += train_abs_errors.sum().item()
                            epoch_mse_sum += train_squared_errors.sum().item()
                            epoch_sample_count += train_graph_1.num_graphs

                    if iterations_training % self.config['log_interval'] == 0:
                        observations_done = (
                            batch_idx + 1) * train_graph_1.num_graphs
                        total_observations = len(self.train_loader.dataset)

                        print(f"[Train] Epoch {epoch} | Batch {batch_idx} "
                              f"| Progress: {observations_done}/{total_observations} "
                              f"| train loss: {train_loss.item():.6f}")

                        self.writer.add_scalar(
                            'train/loss', train_loss.item(), iterations_training)
                        self.writer.add_scalar(
                            'train/lr', optimizer.param_groups[0]['lr'], iterations_training)
                        self.writer.add_scalar(
                            'train/epoch', epoch, iterations_training)

                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()

                    iterations_training += 1

                epoch_mae = epoch_mae_sum / epoch_sample_count
                epoch_mse = epoch_mse_sum / epoch_sample_count

                print(
                    f"[Train] Epoch {epoch} | MAE: {epoch_mae:.6f} | MSE: {epoch_mse:.6f}")
                self.writer.add_scalar('train/MAE', epoch_mae, epoch)
                self.writer.add_scalar('train/MSE', epoch_mse, epoch)

                # validation
                if iterations_training % self.config['validation_interval'] == 0:
                    val_mae_sum = 0.0
                    val_mse_sum = 0.0
                    val_sample_count = 0

                    with torch.no_grad():
                        model.eval()

                        for val_batch_idx, (val_graphs, val_targets, _) in enumerate(self.valid_loader):
                            val_graphs = val_graphs.to(self.device)

                            if self.config["task_type"] == "regression":
                                norm_val_targets = self.normalizer.norm(
                                    val_targets).to(self.device)
                            elif self.config["task_type"] == "classification":
                                raise NotImplementedError(
                                    "Classification task not implemented yet")
                            val_output = model(val_graphs).squeeze(-1)

                            val_loss = self.criterion(
                                val_output, norm_val_targets)

                            if self.config["task_type"] == "regression":
                                val_output_denorm = self.normalizer.denorm(
                                    val_output)
                                val_targets_denorm = self.normalizer.denorm(
                                    norm_val_targets)
                                val_abs_errors = torch.abs(
                                    val_output_denorm - val_targets_denorm)
                                val_squared_errors = (
                                    val_output_denorm - val_targets_denorm) ** 2

                                val_mae_sum += val_abs_errors.sum().item()
                                val_mse_sum += val_squared_errors.sum().item()
                                val_sample_count += val_graphs.num_graphs

                            if validation_iterations % self.config['val_log_interval'] == 0:
                                print(f"Validation Epoch: {epoch} [{val_batch_idx * len(val_graphs)}/{len(self.valid_loader.dataset)}] "
                                      f"Loss: {val_loss.item():.6f}")

                                self.writer.add_scalar(
                                    'validation/loss', val_loss.item(), validation_iterations)
                                self.writer.add_scalar(
                                    'validation/lr', optimizer.param_groups[0]['lr'], validation_iterations)
                                self.writer.add_scalar(
                                    'validation/epoch', epoch, validation_iterations)
                                self.writer.add_scalar(
                                    'train epoch', epoch, validation_iterations)

                            validation_iterations += 1

                    avg_val_mae = val_mae_sum / val_sample_count
                    avg_val_mse = val_mse_sum / val_sample_count

                    print(
                        f"[Validation] Epoch {epoch} | MAE: {avg_val_mae:.6f} | MSE: {avg_val_mse:.6f}")

                    self.writer.add_scalar(
                        'validation/MAE', avg_val_mae, epoch)
                    self.writer.add_scalar(
                        'validation/MSE', avg_val_mse, epoch)

                    if avg_val_mae < best_val_mae:
                        print("Better validation MAE found.")
                        best_epoch_for_val = epoch
                        best_val_mae = avg_val_mae

                        torch.save(model.state_dict(), os.path.join(
                            model_checkpoint_folder, f'best_model_epoch_{epoch}.pth'))

                        print(f"Best model saved with MAE: {avg_val_mae:.6f}")
                        self.writer.add_scalar(
                            'validation/best_loss_MAE', best_val_mae, epoch)
                        self.writer.add_scalar(
                            'validation/best_epoch', epoch, epoch)
                        self.writer.add_scalar(
                            'validation/best_lr', optimizer.param_groups[0]['lr'], epoch)

                    print(f" Validation MSE this epoch: {avg_val_mse:.6f}, "
                          f" MAE this epoch: {avg_val_mae:.6f}, Best so far MAE: {best_val_mae:.6f}, At epoch: {best_epoch_for_val}")

                scheduler.step()
        # save final model
        torch.save(model.state_dict(), os.path.join(
            model_checkpoint_folder, 'final_model.pth'))

        if self.config["do_testing"]:
            print("Testing")
            total_test_mae_sum = 0.0
            total_test_mse_sum = 0.0
            total_test_count = 0.0
            all_predictions = []
            all_targets = []
            all_predictions_id = []

            with torch.no_grad():
                model.eval()

                for test_batch_idx, (test_graphs, test_targets, prediction_id) in enumerate(self.test_loader):
                    test_graphs = test_graphs.to(self.device)

                    if self.config["task_type"] == "regression":
                        norm_test_targets = self.normalizer.norm(
                            test_targets).to(self.device)
                    elif self.config["task_type"] == "classification":
                        raise NotImplementedError(
                            "Classification task not implemented yet")

                    test_output = model(test_graphs).squeeze(-1)

                    if self.config["task_type"] == "regression":
                        test_output_denorm = self.normalizer.denorm(
                            test_output)
                        test_targets_denorm = self.normalizer.denorm(
                            norm_test_targets)

                        test_abs_errors = torch.abs(
                            test_output_denorm - test_targets_denorm)
                        test_squared_errors = (
                            test_output_denorm - test_targets_denorm) ** 2

                        total_test_mae_sum += test_abs_errors.sum().item()
                        total_test_mse_sum += test_squared_errors.sum().item()
                        total_test_count += test_graphs.num_graphs
                    else:
                        raise NotImplementedError(
                            "Classification task not implemented yet")

                    all_predictions.append(test_output_denorm.cpu().numpy())
                    all_targets.append(test_targets_denorm.cpu().numpy())
                    all_predictions_id.append(prediction_id)

                avg_test_mae = total_test_mae_sum / total_test_count
                avg_test_mse = total_test_mse_sum / total_test_count

                print(
                    f"[Test] MAE: {avg_test_mae:.6f} | MSE: {avg_test_mse:.6f}")
                self.writer.add_scalar('test/MAE', avg_test_mae)
                self.writer.add_scalar('test/MSE', avg_test_mse)

                all_predictions = np.concatenate(all_predictions, axis=0)
                all_targets = np.concatenate(all_targets, axis=0)
                all_predictions_id = np.concatenate(all_predictions_id, axis=0)

                df_test = pd.DataFrame({
                    "id": all_predictions_id,
                    "prediction": all_predictions.flatten(),
                    "target": all_targets.flatten()
                })
                df_test.to_csv(os.path.join(model_checkpoint_folder,
                               'test_predictions.csv'), index=False)

                with open(os.path.join(model_checkpoint_folder, 'test_logs_of_predictions.json'), 'w') as f:
                    json.dump({
                        "test_mae_total": total_test_mae_sum,
                        "test_mse_total": total_test_mse_sum,
                        "test_count": total_test_count,
                        "test_mae_avg": avg_test_mae,
                        "test_mse_avg": avg_test_mse
                    }, f)
        end_training_time = time.time()

        elapsed_time = end_training_time - start_training_time

        if self.config["do_training"]:
            elapsed_time_str = time.strftime(
                "%H:%M:%S", time.gmtime(elapsed_time))
            print(f"Training time: {elapsed_time_str}")
            with open(os.path.join(model_checkpoint_folder, 'runtime_and_best_epoch.txt'), 'w') as f:
                f.write(f"Training time: {elapsed_time_str}\n")
                f.write(f"Best epoch for validation: {best_epoch_for_val}\n")
                f.write(f"Best validation MAE: {best_val_mae:.6f}\n")


if __name__ == "__main__":
    config_path = input(
        "Enter the path to your config_finetune.yaml file: ").strip()

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["config_file_path"] = config_path

    if "saved_model_path" not in config or not config["saved_model_path"]:
        saved_model_path = input("Enter the path to the saved model: ").strip()
        print("saved_model_path", saved_model_path)
        config["saved_model_path"] = saved_model_path
    else:
        print("using saved_model_path from config")
        print("saved_model_path", config["saved_model_path"])
    print(config)

    contrastive_run = Finetune(config)
    contrastive_run.train()

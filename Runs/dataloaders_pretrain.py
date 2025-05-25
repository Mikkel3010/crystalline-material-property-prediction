from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
import torch
from torch_geometric.data import Batch
random_seed = 123
torch.manual_seed(random_seed)
np.random.seed(random_seed)


def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_val_ratio=(0.95, 0.05, None), num_workers=5,
                              pin_memory=False):

    total_size = dataset.num_graphs
    train_ratio, val_ratio, _ = train_val_ratio

    indices = list(range(total_size))

    np.random.shuffle(indices)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    # test_sampler = SubsetRandomSampler(test_indices)
    # train_loader = DataLoader(dataset, batch_size=batch_size,
    #                           sampler=train_sampler,
    #                           num_workers=num_workers, drop_last=True,
    #                           collate_fn=collate_fn, pin_memory=pin_memory)
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler, collate_fn=collate_fn,
                              num_workers=num_workers, drop_last=True, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler, collate_fn=collate_fn,
                            num_workers=num_workers, drop_last=True, pin_memory=pin_memory)
    # test_loader = DataLoader(dataset, batch_size=batch_size,
    #                          sampler=test_sampler, collate_fn=collate_fn,
    #                          num_workers=num_workers, drop_last=True, pin_memory=pin_memory)
    # if return_test:
    #     test_loader = DataLoader(dataset, batch_size=batch_size,
    #                              sampler=test_sampler,
    #                              num_workers=num_workers, drop_last=True,
    #                              collate_fn=collate_fn, pin_memory=pin_memory)
    # if return_test:
    #     return train_loader, val_loader, test_loader
    # else:
    #     return train_loader, val_loader
    return train_loader, val_loader


# Batch.from_data_list


def custom_collate_fn(batch):
    data1_list, data2_list, idx_list = zip(*batch)  # unzip the batch of tuples

    batch1 = Batch.from_data_list(data1_list)
    batch2 = Batch.from_data_list(data2_list)
    idx_tensor = torch.tensor(idx_list)
    return batch1, batch2, idx_tensor


# def collate_pool(data_list):
#     """
#     Collate a list of data and return a batch for predicting crystal
#     properties.
#     Parameters
#     ----------
#     dataset_list: list of tuples for each data point.
#       (atom_fea, nbr_fea, nbr_fea_idx)
#       atom_fea: torch.Tensor shape (n_i, atom_fea_len)
#       nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
#       nbr_fea_idx: torch.LongTensor shape (n_i, M)
#       cif_id: str or int
#     Returns
#     -------
#     N = sum(n_i); N0 = sum(i)
#     batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
#       Atom features from atom type
#     batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
#       Bond features of each atom's M neighbors
#     batch_nbr_fea_idx: torch.LongTensor shape (N, M)
#       Indices of M neighbors of each atom
#     crystal_atom_idx: list of torch.LongTensor of length N0
#       Mapping from the crystal idx to atom idx
#     batch_cif_ids: list
#     """
#     batch_atom_fea_rot_1, batch_nbr_fea_rot_1, batch_nbr_fea_idx_rot_1 = [], [], []
#     batch_atom_fea_rot_2, batch_nbr_fea_rot_2, batch_nbr_fea_idx_rot_2 = [], [], []
#     crystal_atom_idx = []
#     batch_cif_ids = []
#     base_idx = 0
#     for i, ((atom_fea_rot_1, nbr_fea_rot_1, nbr_fea_idx_rot_1), (atom_fea_rot_2, nbr_fea_rot_2, nbr_fea_idx_rot_2), cif_id) \
#             in enumerate(data_list):
#         n_i = atom_fea_rot_1.shape[0]  # number of atoms for this crystal
#         batch_atom_fea_rot_1.append(atom_fea_rot_1)
#         batch_atom_fea_rot_2.append(atom_fea_rot_2)
#         batch_nbr_fea_rot_1.append(nbr_fea_rot_1)
#         batch_nbr_fea_rot_2.append(nbr_fea_rot_2)
#         batch_nbr_fea_idx_rot_1.append(nbr_fea_idx_rot_1+base_idx)
#         batch_nbr_fea_idx_rot_2.append(nbr_fea_idx_rot_2+base_idx)
#         new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
#         crystal_atom_idx.append(new_idx)
#         batch_cif_ids.append(cif_id)
#         base_idx += n_i
#     return (torch.cat(batch_atom_fea_rot_1, dim=0),
#             torch.cat(batch_nbr_fea_rot_1, dim=0),
#             torch.cat(batch_nbr_fea_idx_rot_1, dim=0),
#             crystal_atom_idx), \
#         (torch.cat(batch_atom_fea_rot_2, dim=0),
#          torch.cat(batch_nbr_fea_rot_2, dim=0),
#          torch.cat(batch_nbr_fea_idx_rot_2, dim=0),
#          crystal_atom_idx), \
#         batch_cif_ids

# config = {
#     "batch_size": 128,
#     "epochs": 15,
#     "lr": 0.00001,
#     "weight_decay": 1e-4,
#     "log_interval": 2,
#     "validation_interval": 2,
#     "val_log_interval": 2,

#     "lambda": 0.0051,

#     "dataset": {
#         "h5_path": r"C:\Users\mikke\Desktop\P6_new\dataset_cachetest\matminer\6e23a3a0bf630a2fba1643d68ecf19c3.h5"
#     },

#     "dataloader": {
#         "train_ratio": 0.95,
#         "val_ratio": 0.05,
#         "test_ratio": None,
#         "num_workers": 4,
#         "seed": 123
#     },

#     "model_params": {
#         "node_dim": 72,
#         "node_expand_dim": 128,
#         "edge_dim": 41,
#         "edge_expand_dim": 128,
#         "MLP1_dim": 128,
#         "MLP2_dim": 128,
#         # should be a list of [in_dim, out_dim] pairs
#         "dimensionslist": (128, 256), (256, 256), (256, 128),  # should be a list of [in_dim, out_dim] pairs
#     }
# }

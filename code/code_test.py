import torch
from rdkit import Chem
from torch_geometric.data import Data
from main import data_load_csv, path

slice_size = 100000 

target_propertys = [
        "smiles",
        "mw",
        "xlogp",
    ]
dataset = data_load_csv(path, target_propertys, slice_size)

ys = torch.tensor([data[1:] for data in dataset], dtype=torch.float)
mean = ys.mean(dim=0)
std  = ys.std(dim=0)
y = (ys - mean) / std


train_ratio = 0.8
split_idx = int(train_ratio*len(dataset))

print(y[:split_idx, :].shape)
print(y[split_idx:, :].shape)
print(split_idx)

print(y)

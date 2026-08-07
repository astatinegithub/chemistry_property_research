from rdkit import Chem
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


from isonet.utils.path import str2path
from isonet.config import ROOT
from isonet.model.dmpnn import *
from isonet.data.dataset import SSLDataset 

import isonet
print(isonet.__file__)

graph_path = ROOT + "data/preprocessed/train_graph.pt"


batch_size = 64
epochs = 10
lr = 1e-3

# dataset
graphs = torch.load(graph_path,weights_only=False)


dataset = SSLDataset(
    graphs,
    mask_ratio=0.15
)

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SSLModel(
    atom_dim=36,      # 현재 x feature 차원
    bond_dim=7,       # bond feature 차원
    hidden_dim=256,
    num_atom_types=11 # unknown 포함
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr
)
criterion = nn.CrossEntropyLoss()


model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        target = batch.atom_target

        pred = model(batch, depth=5)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}",total_loss / len(train_loader))
    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "head": model.head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        },
        ROOT + f"model/checkpoint/ssl_checkpoint_{epoch}epoch.pt"
    )
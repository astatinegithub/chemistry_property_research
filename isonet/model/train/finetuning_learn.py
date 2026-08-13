from rdkit import Chem
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


from isonet.utils.path import str2path
from isonet.config import ROOT
from isonet.model.dmpnn import IsonetModel

torch.manual_seed(25)


train_graph_path = ROOT + ""
valid_graph_path = ROOT + ""


batch_size = 64
epochs = 10
lr = 1e-3

# dataset
train_graphs = torch.load(train_graph_path, weights_only=False)
valid_graphs = torch.load(valid_graph_path, weights_only=False)


train_loader = DataLoader(
    train_graphs,
    batch_size=batch_size,
    shuffle=True
)

valid_loader = DataLoader(
    valid_graphs,
    batch_size=batch_size,
    shuffle=True
)


endpoint = []


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IsonetModel(
    atom_dim=36,      # atom feature 차원
    bond_dim=7,        # bond feature 차원
    dmpnn_hidden_dim=256,
    admet_hidden_dim=256,
    num_endpoint=len(endpoint),
    depth=5
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr
)
criterion = nn.MSELoss(reduction="none")


checkpoint = torch.load(
    ROOT + "model/checkpoint/ssl_checkpoint.pt",
    map_location=device,
    weights_only=False
)
model.encoder.load_state_dict(
    checkpoint["encoder"]
)


model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        pred = model(batch)

        loss_raw = criterion(
            pred,
            batch.y
        )
        mask = batch.y_mask.float()
        loss = (
            loss_raw * mask
        ).sum() / mask.sum().clamp(min=1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | loss {total_loss/ len(train_loader)}", )
    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "head": model.head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        },
        ROOT + f"model/checkpoint/ssl_checkpoint_{epoch}epoch.pt"
    )
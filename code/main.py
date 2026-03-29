import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_add_pool
import torch.optim as optim

import pandas as pd
import json
from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader


from rdkit import Chem
from tqdm.auto import tqdm
# from utils import create_dataloader




# setting a hyperparameter
cfg = {
    "hop_count": 3,
    "epoch_count": 20
}

# gpu setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# path
path = "data/processed_dataset.csv"



class MoleculeDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__()
        self.data, self.slices = self.collate(data_list)



def create_dataloader(dataset, batch_size, IsShffle=True) -> DataLoader:
    dataset = [mol_to_graph(data[0], list(data[1:])) for data in dataset]
    dataset = MoleculeDataset(dataset)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=IsShffle
    )
    return data_loader


def data_load_csv(path: str, targets: list[str], slice_size:int) -> list:
    df = pd.read_csv(path)
    dataset = df[targets].values.tolist()[:slice_size]
    return dataset


def data_load_json(path: str, targets: list[str]) -> list:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    data = pd.DataFrame(data)
    df = data[data["xlogp"] != ""]
    df = df.astype({"mw": "float64", "xlogp":"float64"})
    mask = df["smiles"].map(lambda s: Chem.MolFromSmiles(s) is not None)

    df = df[mask]
    df = df[target_propertys].values.tolist()

    return df


def mol_to_graph(smiles: str, y: list) -> Data:
    mol = Chem.MolFromSmiles(smiles)

    node_feature: list 
    edge_attr  = []
    edge_index = []


    node_feature = [[
    atom.GetAtomicNum(),
    atom.GetDegree(),
    atom.GetFormalCharge(),
    int(atom.GetIsAromatic())
        ]
          for atom in mol.GetAtoms()]


    for bond in mol.GetBonds():
        bond: Chem.rdchem.Bond
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_feature = [
            int(bond.GetBondTypeAsDouble()),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing())
        ]

        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bond_feature)
        edge_attr.append(bond_feature)


    return Data(
        x=torch.tensor(node_feature, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        y=torch.tensor(y, dtype=torch.float).view(1, -1),
    )



class DMPNN(MessagePassing):
    def __init__(self, atom_dim, bond_dim, hidden_dim,
                 W_bias=False):
        super().__init__()

        self.W_i = nn.Linear(atom_dim + bond_dim, hidden_dim, bias=W_bias)
        self.W_m = nn.Linear(hidden_dim, hidden_dim, bias=W_bias)
        self.W_a = nn.Linear(atom_dim + hidden_dim, hidden_dim, bias=W_bias)


    def forward(self, x: Tensor, edge_index: Tensor,
                 edge_attr: Tensor, cfg: dict):
        h = torch.cat([x[edge_index[0]], edge_attr], dim=1)
        h = self.W_i(h)
        h = torch.nn.functional.relu(h)

        for _ in range(cfg["hop_count"]):
            h = self.propagate(edge_index, h=h)
        node_emb = torch.zeros(x.size(0), h.size(1), device=x.device)

        node_emb.index_add_(0, edge_index[1], h)

        node_emb = torch.cat([x, node_emb], dim=1)
        node_emb = self.W_a(node_emb)
        node_emb = torch.relu(node_emb)

        return node_emb
    

    def message(self, h_j):
        h = self.W_m(h_j)
        h = torch.nn.functional.relu(h)
        return h



class FeedForward(nn.Module):
    def __init__(self, in_dim, drop_rate):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(in_dim, 4*in_dim),
        nn.ReLU(),
        nn.Dropout(drop_rate),
        nn.Linear(4*in_dim, in_dim),
        )
    
    def forward(self, x):
        return self.layers(x)



class ChemModel(nn.Module):
    def __init__(self, in_dim, out_dim, drop_rate=0.2):
        super().__init__()
        self.gnn = DMPNN(
            atom_dim=4,
            bond_dim=3,
            hidden_dim=in_dim
        )

        self.ffn = FeedForward(in_dim, drop_rate)

        self.last_layer = nn.Linear(in_dim, out_dim)


    def forward(self, data: Data, cfg):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        h = self.gnn(x, edge_index, edge_attr, cfg)
        h = global_add_pool(h, batch)
        h
        h = self.ffn(h)
        h = self.last_layer(h)

        return h



if __name__ == "__main__":
    target_propertys = [
        "smiles",
        "mw",
        "xlogp",
    ]
    


    slice_size = 600000 
    dataset = data_load_csv(path, target_propertys, slice_size)    
    train_ratio = 0.8
    split_idx = int(train_ratio*len(dataset))
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]


    model = ChemModel(
        in_dim=64,
        out_dim=len(target_propertys)-1 
    )
    model = model.to(device)


    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()


    train_loader = create_dataloader(train_data, batch_size=32) 


    for _ in range(cfg["epoch_count"]):
        loader = train_loader
        for batch in tqdm(loader):
            batch = batch.to(device) # 배치 GPU사용 가능하게 만들기
            pred = model(batch, cfg)
            loss = loss_fn(pred, batch.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), 'save/model.pth')


    model.eval()
    test_loader = create_dataloader(test_data, batch_size=32)


    with torch.no_grad():
        total_loss = 0
        loader = test_loader
        for batch in tqdm(loader):
            batch = batch.to(device) # 배치 GPU사용 가능하게 만들기
            pred = model(batch, cfg)
            loss = loss_fn(pred, batch.y)

            total_loss += loss.item()

    print("Test Loss:", total_loss)

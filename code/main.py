import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.data import Data
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader


import pandas as pd
import json
import time
from rdkit import Chem
from tqdm.auto import tqdm
import os


from models import ChemModel


# setting a hyperparameter
cfg = {
    "depth": 3,
    "epoch_count": 20,
    "data_path": "data/processed_dataset_5_property.csv",
    "save_path": "Model/", # "model_loss_000.pth"
    "in_dim": 256
}

# gpu setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# path setting
path = cfg["data_path"]



class MoleculeDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__()
        self.data, self.slices = self.collate(data_list)



def create_dataloader(dataset, mean, std ,batch_size, IsShffle=True) -> DataLoader:
    dataset = [
        mol_to_graph(data[0], list(data[1:]), mean, std) 
        for data in dataset
    ]

    dataset = MoleculeDataset(dataset)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=IsShffle
    )
    return data_loader



def data_load_csv(path: str, targets: list[str], slice_size:int) -> list:
    df = pd.read_csv(path)
    smiles = ["smile", "SMILE", "smiles", "SMILES"]


    if any(smile in df.keys().tolist() for smile in smiles):
        key = df.keys()[1] if df.keys()[1] in smiles else df.keys()[0]
    else:
        raise "smile를 포함하고있지 않은 데이터 셋이거나, smile의 컬럼명이 smile, smiles, SMILE, SMILES가 아닙니다."
    print(f"smiles colmun name : {key}")


    # mask = df[key].map(lambda s: Chem.MolFromSmiles(s) is not None)
    # df = df[mask]


    if slice_size in [None, "all"]:
        dataset = df[targets].values.tolist()
    else:
        dataset = df[targets].values.tolist()[:slice_size]
    return dataset


def data_load_json(path: str, targets: list[str]) -> list:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    data = pd.DataFrame(data)
    df = data[data["xlogp"] != ""]
    df = df.astype({p: "float64" for p in targets})
    mask = df["smiles"].map(lambda s: Chem.MolFromSmiles(s) is not None)

    df = df[mask]
    df = df[target_propertys].values.tolist()

    return df



def build_reverse_edge_index(edge_index: list) -> list:

    edge_dict = {}
    for i, (s, d) in enumerate(edge_index):
        edge_dict[(s, d)] = i

    rev_edge = []
    for s, d in edge_index:
        rev_edge.append(edge_dict[(d, s)])

    return rev_edge



def mol_to_graph(smiles: str, y: list, mean: Tensor, std: Tensor) -> Data:
    mol = Chem.MolFromSmiles(smiles)

    node_feature = []
    edge_attr    = []
    edge_index   = []


    node_feature = [[
            atom.GetAtomicNum(), # 원자번호
            atom.GetDegree(), # 결합수
            atom.GetFormalCharge(), # 형식전하
            atom.GetTotalNumHs(), # 생략된 붙어있는 H의 수
            atom.GetMass(), # 원자 질량
            int(atom.GetIsAromatic()) 
        ]
          for atom in mol.GetAtoms()
    ]


    for bond in mol.GetBonds():
        bond: Chem.rdchem.Bond
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_feature = [
            int(bond.GetBondTypeAsDouble()),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()) #고리여부
        ]

        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bond_feature)
        edge_attr.append(bond_feature)


    y = (torch.tensor(y, dtype=torch.float) - mean) / std
    rev_edge = build_reverse_edge_index(edge_index)


    return Data(
        x=torch.tensor(node_feature, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        rev_edge=torch.tensor(rev_edge, dtype=torch.long),
        y=y.view(1, -1),
    )



def fit(model, data_loader, optimizer, loss_fn, cfg, device) -> list:
    # train_loss = []
    for i in range(cfg["epoch_count"]):
        for batch in tqdm(data_loader, desc=f"{i+1} 번째 epoch "):
            batch = batch.to(device) # 배치 GPU사용 가능하게 만들기
            pred = model(batch, cfg)
            loss = loss_fn(pred, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 코랩 런타임 끊어짐 방지용
        torch.save({
            "model": model.state_dict(),
            "mean": mean,
            "std": std,
            "in_dim": cfg["in_dim"]
        }, cfg["save_path"] + f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_epoch{i+1}.pth")


        # train_loss.append(loss.item())
    # return train_loss


def train_weight_load(target_dir: str, device):
    files = os.listdir(target_dir)
    if len(files) == 0:
        return None
    else:
        files = [f for f in files if f.endswith(".pth")]
        target_file = sorted(files)[-1]
        checkpoint = torch.load(target_dir+'/'+target_file, map_location=device) 
        
        return checkpoint


# def init_data_target_make():



if __name__ == "__main__":

    target_propertys = [
        "SMILES",
        "Molecular_Weight",
        "XLogP",
        # "Charge",
        "Polar_Area"
    ]
    

    slice_size = "all"
    dataset = data_load_csv(path, target_propertys, slice_size)


    train_ratio = 0.8
    split_idx = int(train_ratio*len(dataset))
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]


    model = ChemModel(
        in_dim=cfg["in_dim"],
        out_dim=len(target_propertys)-1 
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # 재학습시
    checkpoint = train_weight_load(cfg["save_path"], device=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    mean = checkpoint["mean"]
    std = checkpoint["std"]

    # 아래 주석처리하고 재학습
    ys = torch.tensor([data[1:] for data in dataset], dtype=torch.float)
    mean = ys.mean(dim=0)
    std  = ys.std(dim=0)
    std[std < 1e-6] = 1.0
    print("std :", std)

    cfg["mean"] = mean
    cfg["std"] = std

    
    model.train()  


    train_loader = create_dataloader(train_data, mean, std, batch_size=32)     
    fit(model, train_loader, optimizer, loss_fn, cfg, device)    


    torch.save({
        "model": model.state_dict(),
        "mean": mean,
        "std": std,
        "in_dim": cfg["in_dim"],
        "optimizer": optimizer.state_dict()
    }, cfg["save_path"])


    model.eval()
    test_loader = create_dataloader(test_data, mean, std, batch_size=32)


    with torch.no_grad():
        total_loss = 0
        loader = test_loader
        for batch in tqdm(loader):
            batch = batch.to(device) # 배치 GPU사용 가능하게 만들기
            pred = model(batch, cfg)
            loss = loss_fn(pred, batch.y)

            total_loss += loss.item()

    print("Test Loss:", total_loss)

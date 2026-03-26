import pandas as pd
from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
import torch
from rdkit import Chem


# target_propertys = ["smiles", "Molecular Weight"]


# data_path="data/delaney-processed.csv"
# df = pd.read_csv(data_path)
# dataset = df[target_propertys].values.tolist()
# for i in dataset:
#     print(i)
target_propertys = [
        "smiles",
        "mw",
        "xlogp",
    ]
data_path="data/delaney-processed.csv"
df = pd.read_csv(data_path)
# print(df.info())
# print(target_propertys[0],target_propertys[1:])
import json

with open("data/PubChem_compound_ethanol.json", "r", encoding="utf-8") as file:
    data = json.load(file)
data = pd.DataFrame(data)
# print(data.info())
# print(data[data["xlogp"] == ""]) # -> 이놈없는놈만 129629개임
# print(data[data["smiles"] == ""])

df = data[data["xlogp"] != ""]
df = df.astype({"mw": "float64", "xlogp":"float64"})
# print(df.info())
df = df.astype({"mw": "float64", "xlogp":"float64"})

mask = df["smiles"].map(lambda s: Chem.MolFromSmiles(s) is not None)

df = df[mask]
df.to_csv("data/processed_dataset.csv")
# df = df[target_propertys].values.tolist()


# for i, (m, x, y) in enumerate(dataset):
#     if m=="C=BrC1=CC=C(C=C1)[C@@H](C(F)(F)F)O":
#         print(i, x, y, m)

# test_idx = 31369
# print(dataset[test_idx][0], dataset[test_idx][1:])
# valid_data = [d for d in dataset if Chem.MolFromSmiles(d[0]) is not None]
# ds = dataset[~dataset["smiles"].isin(valid_data)]
# print(ds.info())
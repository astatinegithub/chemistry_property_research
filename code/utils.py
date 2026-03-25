import pandas as pd
from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
import torch


# target_propertys = ["smiles", "Molecular Weight"]


# data_path="data/delaney-processed.csv"
# df = pd.read_csv(data_path)
# dataset = df[target_propertys].values.tolist()
# for i in dataset:
#     print(i)
target_propertys = [
        "smiles",
        "Molecular Weight",
        "ESOL predicted log solubility in mols per litre",
        "Number of Rings"
    ]
data_path="data/delaney-processed.csv"
df = pd.read_csv(data_path)
# print(df.info())
print(target_propertys[0],target_propertys[1:])
import json

with open("data/PubChem_compound_ethanol.json", "r", encoding="utf-8") as file:
    data = json.load(file)
data = pd.DataFrame(data)
print(data.info())
data.dropna()
print(data.info())
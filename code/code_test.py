import torch
from rdkit import Chem
from torch_geometric.data import Data
from main import data_load_csv, path
import pandas as pd



df = pd.read_csv(path)
smiles = ["smile", "SMILE", "smiles", "SMILES"]

print(df.keys())
print(type(df.keys()))
print(df.keys().tolist())
if any(smile in df.keys().tolist() for smile in smiles):
    key = df.keys()[1] if df.keys()[1] in smiles else df.keys()[0]
else:
    raise "smile를 포함하고있지 않은 데이터 셋이거나, smile의 컬럼명이 smile, smiles, SMILE, SMILES가 아닙니다."
print(f"smiles colmun name : {key}")

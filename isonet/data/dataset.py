from rdkit import Chem
import time
from tqdm import tqdm
import numpy as np
import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset

from isonet.utils.path import str2path
from isonet.config import ROOT

ATOM_TYPES = {
    1:0,    # H
    6:1,    # C
    7:2,    # N
    8:3,    # O
    9:4,    # F
    15:5,   # P
    16:6,   # S
    17:7,   # Cl
    35:8,   # Br
    53:9    # I
}
BOND_TYPES = {
    1.0:0,
    2.0:1,
    3.0:2,
    1.5:3
}


def one_hot(value, mapping):
    idx = mapping.get(value, len(mapping))
    return F.one_hot(
        torch.tensor(idx),
        num_classes=len(mapping)+1   # +1 = unknown
    ).float()


def build_reverse_edge_index(edge_index: list) -> list:
    edge_dict = {}
    for i, (s, d) in enumerate(edge_index):
        edge_dict[(s, d)] = i

    rev_edge = []
    for s, d in edge_index:
        rev_edge.append(edge_dict[(d, s)])

    return rev_edge


def mol2feature(mol: Chem.Mol) -> Data:
        node_feature = []
        edge_attr    = []
        edge_index   = []

        
        node_feature = [
            torch.cat([
                one_hot(atom.GetAtomicNum(), ATOM_TYPES), # 원소
                F.one_hot(
                    torch.tensor(atom.GetDegree()),
                    num_classes=7
                ).float(),                                     # 결합 수
                F.one_hot(
                    torch.tensor(atom.GetFormalCharge()+5),
                    num_classes=11
                ).float(),                                    # 전하 -5~5
                F.one_hot(
                    torch.tensor(atom.GetTotalNumHs()),
                    num_classes=5
                ).float(),                                     # H 개수
                torch.tensor([
                    atom.GetMass()/100,
                    int(atom.GetIsAromatic())
                ], dtype=torch.float)
            ])
            for atom in mol.GetAtoms()
        ]   


        for bond in mol.GetBonds():
            bond: Chem.rdchem.Bond
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            bond_feature = torch.cat([
                one_hot(
                    bond.GetBondTypeAsDouble(),
                    BOND_TYPES
                ),
                torch.tensor([
                    int(bond.GetIsConjugated()),
                    int(bond.IsInRing())
                ], dtype=torch.float)
            ])

            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_attr.append(bond_feature)
            edge_attr.append(bond_feature)

        rev_edge = build_reverse_edge_index(edge_index)

        atom_type = torch.tensor(
            [ATOM_TYPES[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
        )


        x = torch.stack(node_feature).to(torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr).float()
        rev_edge = torch.tensor(rev_edge, dtype=torch.long)
        

        return x, edge_index, edge_attr, rev_edge, atom_type



class MolGraph(Data):
    def __init__(self, x=None, edge_index=None,
                edge_attr=None, rev_edge=None,
                atom_type=None, y=None, y_mask=None):
        super().__init__(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            rev_edge=rev_edge,
            atom_type=atom_type,
            y=y,
            y_mask=y_mask
        )

        # for pretrain variables
        self.mask_idx = None
        self.atom_target = None


    def __inc__(self, key, value, *args, **kwargs): # rev_edge는 커스텀이라 batch계산을 위해 필요함
        if key == "rev_edge":
            return self.edge_attr.size(0) # bond 수
        if key == "mask_idx":
            return self.x.size(0) # atom 수
        if key == "atom_target":  # 변하면 안되기에 
            return 0
        return super().__inc__(key, value, *args, **kwargs)




class MoleculeDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]



class SSLDataset(Dataset):
    def __init__(self, data_list, mask_ratio=0.15):
        super().__init__()

        self.data_list = data_list
        self.mask_ratio = mask_ratio


    def len(self):
        return len(self.data_list)


    def get(self, idx):
        data = deepcopy(self.data_list[idx])
        num_atoms = data.x.size(0)
        num_mask = max(1,int(num_atoms * self.mask_ratio))
        mask_idx = torch.randperm(num_atoms)[:num_mask]

        data.atom_target = data.atom_type[mask_idx]
        data.mask_idx = mask_idx

        # masking
        data.x[mask_idx] = 0

        return data


# def create_dataloader(dataset, batch_size, IsPretrain=True, IsShffle=True) -> DataLoader:
#     dataset = [
#         MolGraph(data) for data in tqdm(dataset, desc="loading")
#     ]

#     if IsPretrain:
#         dataset = list(map(mask_atom, dataset))

#     dataset = MoleculeDataset(dataset)

#     data_loader = DataLoader(
#         dataset=dataset,
#         batch_size=batch_size,
#         shuffle=IsShffle
#     )
#     return data_loader


def create_ssl_dataloader(path, batch_size=64,
                          mask_ratio=0.15, shuffle=True):

    graphs = torch.load(path)
    dataset = SSLDataset(graphs, mask_ratio)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return loader



if __name__ == "__main__":
    import time
    smiles = Chem.SDMolSupplier(ROOT+"dataset/raw/Compound_000000001_000500000.sdf")

    # for i, mol in enumerate(smiles):
    #     if mol is None:
    #         continue

    #     if mol.GetNumAtoms() == 1:
    #         print(i, Chem.MolToSmiles(mol))

    # print(len(smiles))
    # T = time.time()
    # train_dataloader = create_dataloader(
    #     dataset=smiles,
    #     batch_size=16
    # )

    mol = next(smiles)
    data = MolGraph(*mol2feature(mol))
    print(data.x.shape)
    print(data.edge_index.shape)
    print(data.edge_attr.shape)
    print(data.rev_edge)
    # print(time.time()-T)
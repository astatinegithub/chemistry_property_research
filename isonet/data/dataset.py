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


# Chemprop v2 기준
ATOM_NUMS = list(range(1, 37)) + [53]  # H ~ Kr + I
DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
CHIRAL_TAGS = [0, 1, 2, 3]
NUM_HS = [0, 1, 2, 3, 4]

HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP2D,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

BOND_STEREOS = [0, 1, 2, 3, 4, 5]


def one_hot_unknown(value, choices):
    """choices + unknown slot"""
    out = torch.zeros(len(choices) + 1)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices)
    out[idx] = 1.0
    return out


def build_reverse_edge_index(edge_index: list) -> list:
    edge_dict = {}
    for i, (s, d) in enumerate(edge_index):
        edge_dict[(s, d)] = i

    rev_edge = []
    for s, d in edge_index:
        rev_edge.append(edge_dict[(d, s)])

    return rev_edge

def atom_feature(atom: Chem.Atom, use_stereo=True) -> Tensor:
    atomic_num = one_hot_unknown(atom.GetAtomicNum(), ATOM_NUMS)              # 1. atomic number: 37 + unknown = 38
    degree = one_hot_unknown(atom.GetDegree(), DEGREES)                       # 2. degree: 6 + unknown = 
    charge = one_hot_unknown(atom.GetFormalCharge(), FORMAL_CHARGES)          # 3. formal charge: 5 + unknown = 6

    if use_stereo:                                                            # 4. chirality: 4 + unknown = 5
        chirality = one_hot_unknown(int(atom.GetChiralTag()), CHIRAL_TAGS)
    else:
        chirality = torch.zeros(5) # 중요: 제거하지 말고 5차원 그대로 0
        
    num_h = one_hot_unknown(atom.GetTotalNumHs(), NUM_HS)                     # 5. H count: 5 + unknown = 6
    hybridization = one_hot_unknown(atom.GetHybridization(), HYBRIDIZATIONS)  # 6. hybridization: 7 + unknown = 8
    aromatic = torch.tensor([float(atom.GetIsAromatic())])                    # 7. aromatic: 1 
    mass = torch.tensor([atom.GetMass() / 100.0])                             # 8. mass: 1

    feature = torch.cat([
        atomic_num,       # 38
        degree,           # 7
        charge,           # 6
        chirality,        # 5
        num_h,            # 6
        hybridization,    # 8
        aromatic,         # 1
        mass,             # 1
    ])

    assert feature.shape[0] == 72

    return feature


def bond_feature(bond: Chem.Bond, use_stereo=True) -> Tensor:
    null = torch.tensor([0.0])    # null bit
    # 4 bond types
    bond_type = torch.tensor([float(bond.GetBondType() == bt) for bt in BOND_TYPES])
    conjugated = torch.tensor([float(bond.GetIsConjugated())])
    ring = torch.tensor([float(bond.IsInRing())])
    # 6 stereo + unknown = 7
    if use_stereo:
        stereo = one_hot_unknown(int(bond.GetStereo()), BOND_STEREOS)
    else:
        stereo = torch.zeros(7)

    feature = torch.cat([
        null,          # 1
        bond_type,     # 4
        conjugated,    # 1
        ring,          # 1
        stereo,        # 7
    ])

    assert feature.shape[0] == 14

    return feature


def mol2feature(mol: Chem.Mol, use_stereo=True) -> Data:
    edge_attr   = []
    edge_index  = []
    
    node_feature = [atom_feature(atom, use_stereo) for atom in mol]

    for bond in mol.GetBonds():
        bond: Chem.rdchem.Bond
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_features = bond_feature(bond, use_stereo)

        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bond_features)
        edge_attr.append(bond_features)

    rev_edge = build_reverse_edge_index(edge_index)
    atom_type = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()])


    x = torch.stack(node_feature).float()
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
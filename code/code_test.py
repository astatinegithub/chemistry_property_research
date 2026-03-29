import torch
from rdkit import Chem
from torch_geometric.data import Data

def mol_to_graph(smiles: str, y: list) -> Data:
    mol = Chem.MolFromSmiles(smiles)

    node_feature: list 
    edge_attr: list
    edge_index: list


    node_feature = [
    [
    atom.GetAtomicNum(),
    atom.GetDegree(),
    atom.GetFormalCharge(),
    int(atom.GetIsAromatic())
    ]
    for atom in mol.GetAtoms()] # 원자갯수가


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

smile = "CCO.C1=CC=NC(=C1)C2=CC=CC=N2.C1=CC=NC(=C1)C2=CC=CC=N2.C1=CC=NC(=C1)C2=CC=CC=N2.C1=CC=NC(=C1)C2=CC=CC=N2.C(#N)[N-]C#N.[O-]Cl(=O)(=O)=O.[O-]Cl(=O)(=O)=O.[O-]Cl(=O)(=O)=O.[Cu+2].[Cu+2]"
mol = Chem.MolFromSmiles(smile)

atoms = [[
    atom.GetAtomicNum(),
    atom.GetDegree(),
    atom.GetFormalCharge(),
    int(atom.GetIsAromatic())
        ]
          for atom in mol.GetAtoms()]
atom_tensor = torch.tensor(atoms)

print(atom_tensor.shape)
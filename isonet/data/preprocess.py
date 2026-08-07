from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm
import torch

from isonet.config import ROOT
from isonet.data.dataset import MolGraph


RDLogger.DisableLog("rdApp.*")
allowed_atoms = {
    1,6,7,8,9,15,16,17,35,53
}

def preprocess_sdf(input_path, output_path):
    Mols = Chem.SDMolSupplier(input_path,removeHs=False)

    graphs = []
    removed = 0
    for i, mol in enumerate(tqdm(Mols, desc="processing", mininterval=0.3)):
        # RDKit 변환 실패
        if mol is None:
            removed += 1
            continue

        if any(
            atom.GetAtomicNum() not in allowed_atoms
            for atom in mol.GetAtoms()
        ):
            continue

        # 결합 없는 원자/이온 제거
        if mol.GetNumBonds() == 0:
            removed += 1
            continue

        if i == 100000:
            break

        
        graph = MolGraph(mol)
        graphs.append(graph)


    torch.save(graphs, output_path)

    print("====================")
    print(f"saved : {len(graphs)}")
    print(f"removed : {removed}")
    print(f"path : {output_path}")
    print("====================")


if __name__ == "__main__":
    input_path = ROOT + "dataset/test_dataset/Compound_000000001_000500000.sdf"
    output_path = ROOT + "dataset/processed_data/train_graph.pt"

    preprocess_sdf(input_path, output_path)


    # smiles = Chem.SDMolSupplier(ROOT+"dataset/test_dataset/Compound_000000001_000500000.sdf")
    
    # mol = smiles[16238]
    # print(Chem.MolToSmiles(mol))

    # for atom in mol.GetAtoms():
    #     print(atom.GetSymbol(), atom.GetDegree())
    
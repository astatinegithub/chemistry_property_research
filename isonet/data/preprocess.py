from rdkit import Chem
from rdkit import RDLogger
from pathlib import Path
from tqdm import tqdm
import torch

from isonet.config import ROOT
from isonet.data.dataset import MolGraph, mol2feature
from isonet.data.dataset_validation import *


allowed_atoms = {
    1,6,7,8,9,15,16,17,35,53
}

def mol2graph(mol, y=None, y_mask=None): # 수정중
    # RDKit 변환 실패
    if mol is None:
        return None
    
    graph = MolGraph(*mol2feature(mol))

    if y is not None:
        graph.y = y
    if y_mask is not None:
        graph.y_mask = y_mask

    return graph


class SDFReader:
    def __init__(self, input_path):
        self.mols = Chem.SDMolSupplier(input_path, removeHs=False)


    def __len__(self):
        return len(self.mols)


    def __iter__(self):
        for mol in self.mols:
            yield mol, None
    


def makeMolGraph(reader, output_path, validator: MolValidator, max_len=None) -> list:
    graphs = []
    removed = 0
    with RDKitLogCapture() as log:
        for mol, target in tqdm(reader, desc="processing", mininterval=0.3):
            if (max_len is not None) and (len(graphs) > max_len):
                break 
            logs = log.get()

            if not validator.validate(mol, logs):
                continue


            if target is None:
                graph = mol2graph(mol)
            else: 
                y, y_mask = target
                graph = mol2graph(mol, y, y_mask)


            if graph == None:
                removed += 1
            else:
                graphs.append(graph)

    torch.save(graphs, output_path)

    print("====================")
    print(f"saved : {len(graphs)}")
    validator.report()
    print(f"path : {output_path}")



if __name__ == "__main__":
    input_path = ROOT + "dataset/raw/Compound_000000001_000500000.sdf"
    output_path = ROOT + "dataset/processed_data/clean_dataset.pt"

    reader = SDFReader(input_path)
    validator = MolValidator(allowed_atoms)
    

    makeMolGraph(reader, output_path, validator)


        # print(i, Chem.MolToSmiles(mol))
        # print(mol.GetNumBonds())
    # mol = smiles[221]
    # print(Chem.MolToSmiles(mol))
    # print(smiles[22715])

    
    
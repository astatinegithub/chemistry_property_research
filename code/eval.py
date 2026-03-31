import torch
from rdkit import Chem
from torch_geometric.data import Data

from main import ChemModel

# -----------------------
# 1. 디바이스 설정
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mol_to_graph(smiles: str) -> Data:
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
        edge_attr=torch.tensor(edge_attr, dtype=torch.float)
    )



model = ChemModel(in_dim=64, out_dim=4)

checkpoint = torch.load("Model/model_loss_94.pth", map_location=device)
model.load_state_dict(checkpoint["model"])

model = model.to(device)
model.eval()

cfg = {"hop_count": 3}

mean = checkpoint["mean"]
std = checkpoint["std"]

# -----------------------
# 3. 예측 함수
# -----------------------
def predict(smiles: str):
    data = mol_to_graph(smiles)

    # 🔥 PyG batch 처리 (필수)
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)

    data = data.to(device)

    with torch.no_grad():
        pred = model(data, cfg)

    return pred

# -----------------------
# 4. 실행
# -----------------------
if __name__ == "__main__":
    while True:
        smiles = input("SMILES 입력 (종료: q): ")

        if smiles == "q":
            break

        try:
            pred = predict(smiles)
            pred = pred * std + mean

            # print("예측 결과 → {}".format( for i, p in enumerate(pred[0])))

            # print(f"예측 결과 → MW: {pred[0, 0]:.2f}, XlogP: {pred[0, 1]:.2f}")
            print(pred)

        except Exception as e:
            print("에러:", e)
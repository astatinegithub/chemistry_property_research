import torch
from torch import Tensor
import torch.nn as nn
from rdkit import Chem
from utils import create_dataloader


# setting a hyperparameter
cfg = {
    "mpnn_hop": 3,
    "epoch_count": 50
}


def mol_to_graph(smiles: str) -> Tensor:
    mol = Chem.MolFromSmiles(smiles)

    node_feature = []
    edge_feature = []
    edge_index   = []


    for atom in mol.GetAtoms():
        atom: Chem.rdchem.Atom

        node_feature.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic())
        ])
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
        edge_index.append([i, j])
        edge_feature.append(bond_feature)
        edge_feature.append(bond_feature)
    return (
        torch.tensor(node_feature, dtype=torch.float),
        torch.tensor(edge_feature, dtype=torch.float),
        torch.tensor(edge_index, dtype=torch.long).t(),
    )



class DMPNN(nn.Module):
    def __init__(self, atom_dim, bond_dim, hidden_dim,
                 W_bias=False):
        super().__init__()

        self.W_i = nn.Linear(atom_dim + bond_dim, hidden_dim, bias=W_bias)
        self.W_m = nn.Linear(hidden_dim, hidden_dim, bias=W_bias)
        self.W_a = nn.Linear(atom_dim + hidden_dim, hidden_dim, bias=W_bias)


    def forward(self, node_feature, edge_featrue, edge_index, cfg):
        start_nodes, end_nodes = edge_index

        h = torch.cat([node_feature[start_nodes], edge_featrue], dim=1)
        h = torch.nn.functional.relu(self.W_i(h))

        for _ in range(cfg["mpnn_hop"]):
            new_h = []

            for i in range(h.size(0)):
                v = start_nodes[i]
                w = end_nodes[i]

                incoming = (end_nodes == v) & (start_nodes != w)

                if incoming.sum() == 0:
                    m = torch.zeros_like(h[i])
                else:
                    m = h[incoming].sum(dim=0)

                new_h.append(torch.nn.functional.relu(self.W_m(m)))

            h = torch.stack(new_h)

            node_emb = []
            for v in range(x.size(0)):
                incoming = (end_nodes == v)
                if incoming.sum() == 0:
                    m = torch.zeros(h.size(1))
                else:
                    m = h[incoming].sum(dim=0)

                hv = torch.cat([x[v], m], dim=0)
                node_emb.append(torch.nn.functional.relu(self.W_a(hv)))

            node_emb = torch.stack(node_emb)

            # molecule embedding
            mol_emb = node_emb.sum(dim=0)
            return mol_emb



class PredictModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn = DMPNN(atom_dim=4, bond_dim=3, hidden_dim=64)

        self.ffn = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
        )

    def forward(self, x, edge_attr, edge_index, cfg):
        model_emb = self.gnn(x, edge_attr, edge_index, cfg)
        return self.ffn(model_emb)


data = [
    ("CCO", 0.5),
    ("CC", 0.2),
    ("CCC", 0.3),
    ("c1ccccc1", 0.8)
]

model = PredictModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(cfg["epoch_count"]):
    total_loss = 0

    for smiles, y in data:
        x, edge_attr, edge_index = mol_to_graph(smiles)

        pred = model(x, edge_attr, edge_index)

        target = torch.tensor([y], dtype=torch.float)

        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

# 4. 테스트 실행

if __name__ == "__main__":
    smiles = "CCO"  # 에탄올

    x, edge_attr, edge_index = mol_to_graph(smiles)

    model = PredictModel()
    out = model(x, edge_attr, edge_index, cfg)

    print("예측값:", out.item())
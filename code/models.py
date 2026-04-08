import torch
from torch import nn
from torch import Tensor
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.data import Data



class DMPNN(MessagePassing):
    def __init__(self, atom_dim, bond_dim, hidden_dim,
                 W_bias=False):
        super().__init__()

        self.W_i = nn.Linear(atom_dim + bond_dim, hidden_dim, bias=W_bias)
        self.W_m = nn.Linear(hidden_dim, hidden_dim, bias=W_bias)
        self.W_a = nn.Linear(atom_dim + hidden_dim, hidden_dim, bias=W_bias)


    def forward(self, x: Tensor, edge_index: Tensor,
                 edge_attr: Tensor, rev_edge: Tensor, cfg: dict):
        h0 = torch.cat([x[edge_index[0]], edge_attr], dim=1) # 모든 src의 atomfeature + bondfeature 
        h0 = self.W_i(h0) # (batch_size만큼의 분자의 결합하는 원자수, hidden_dim)
        h0 = torch.relu(h0)
        h = h0.clone()

        for _ in range(cfg["depth"]):
            m = torch.relu(self.W_m(h))

            node_m = torch.zeros(x.size(0), m.size(1), device=x.device)
            node_m.index_add_(0, edge_index[1], m)
            
            m_edge = node_m[edge_index[0]]
            m_edge = m_edge - m[rev_edge]
            h = h0 + m_edge
            # h = h + self.propagate(edge_index, h=h, rev_edge=rev_edge)

        node_emb = torch.zeros(x.size(0), h.size(1), device=x.device)
        node_emb.index_add_(0, edge_index[1], h)

        node_emb = torch.cat([x, node_emb], dim=1)
        node_emb = self.W_a(node_emb)
        node_emb = torch.relu(node_emb)

        return node_emb
    

    def message(self, h_j): # h_j 는 정해진 값임
        h = self.W_m(h_j)
        h = torch.relu(h)
        return h
    

    def update(self, aggr_out, h, rev_edge):
        # aggr_out: sum of incoming messages
        # h: current edge hidden state

        reverse_h = h[rev_edge]
        new_h = aggr_out - reverse_h

        return new_h



class FeedForward(nn.Module):
    def __init__(self, in_dim, drop_rate):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(in_dim, 4*in_dim),
        nn.ReLU(),
        nn.Dropout(drop_rate),
        nn.Linear(4*in_dim, in_dim),
        )


    def forward(self, x):
        return self.layers(x)



class ChemModel(nn.Module):
    def __init__(self, in_dim, out_dim, drop_rate=0.2):
        super().__init__()
        self.gnn = DMPNN(
            atom_dim=6, # 방금 특성 넣어줌 나중에 수정해야 할듯 
            bond_dim=3,
            hidden_dim=in_dim
        )

        self.ffn = FeedForward(in_dim, drop_rate)

        self.last_layer = nn.Linear(in_dim, out_dim)


    def forward(self, data: Data, cfg: dict):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        rev_edge = data.rev_edge
        batch = data.batch

        h = self.gnn(x, edge_index, edge_attr, rev_edge, cfg)
        h = global_add_pool(h, batch)
        h = self.ffn(h)
        h = self.last_layer(h)

        return h
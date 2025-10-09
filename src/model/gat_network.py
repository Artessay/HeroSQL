import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool, global_max_pool

class NodeFeaturizer(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.linear(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x

class GATBlock(nn.Module):
    def __init__(self, hidden_dim, heads=4):
        super().__init__()

        assert hidden_dim % heads == 0
        self.gat = GATConv(hidden_dim, hidden_dim // heads, heads=heads)
        self.act = nn.ReLU()
    def forward(self, x, edge_index):
        x_new = self.gat(x, edge_index)
        x_new = self.act(x_new)
        return x_new

class GATNet(nn.Module):
    def __init__(
            self, 
            input_dim=1024,
            hidden_dim=128, 
            num_layers=2,
            dropout=0.2,
            heads=4
    ):
        super().__init__()
        self.node_featurizer = NodeFeaturizer(input_dim, hidden_dim, dropout)
        self.convs = nn.ModuleList([
            GATBlock(hidden_dim, heads=heads) for _ in range(num_layers)
        ])
        self.dropout = dropout

    def forward(self, node_features, edge_index, batch):
        x = self.node_featurizer(node_features)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph_embedding = global_add_pool(x, batch)
        return graph_embedding
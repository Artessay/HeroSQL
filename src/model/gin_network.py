
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, global_add_pool


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

class GINBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.conv = GINConv(self.mlp)
        
    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

class GINNet(nn.Module):
    def __init__(
            self, 
            input_dim=1024,
            hidden_dim=128, 
            num_layers=3,
            dropout=0.2
    ):
        super().__init__()

        self.node_featurizer = NodeFeaturizer(input_dim, hidden_dim, dropout)

        self.convs = nn.ModuleList([
            GINBlock(hidden_dim) for _ in range(num_layers)
        ])

        self.dropout = dropout

    def forward(self, node_features, edge_index, batch):
        x = self.node_featurizer(node_features)
        
        for conv in self.convs:
            x_new = conv(x, edge_index)
            x = x_new + x   # residual connection
            x = F.dropout(x, p=self.dropout, training=self.training)

        graph_embedding = global_add_pool(x, batch)
        return graph_embedding
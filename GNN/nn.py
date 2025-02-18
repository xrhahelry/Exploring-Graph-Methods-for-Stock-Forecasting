import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATv2Conv


class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

class GAT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GAT, self).__init__()
        self.conv1 = GATv2Conv(input_size, hidden_size, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_size, hidden_size, edge_dim=1)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.conv1(x, edge_index, edge_attr=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_weight)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

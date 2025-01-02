import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class GAT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GAT, self).__init__()
        self.conv1 = GATv2Conv(input_size, hidden_size, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_size, hidden_size, edge_dim=1)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = global_mean_pool(x, batch)  
        x = self.lin(x)
        return x

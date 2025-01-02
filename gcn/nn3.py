import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class TGCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_gat=True):
        super(TGCN, self).__init__()
        self.conv1 = GATv2Conv(input_size, hidden_size) if use_gat else nn.Linear(input_size, hidden_size)
        self.conv2 = GATv2Conv(hidden_size, hidden_size) if use_gat else nn.Linear(hidden_size, hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.conv1(x, edge_index, edge_weight) if isinstance(self.conv1, GATv2Conv) else self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight) if isinstance(self.conv2, GATv2Conv) else self.conv2(x)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

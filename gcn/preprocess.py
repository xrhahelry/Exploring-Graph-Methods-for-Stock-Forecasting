import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx


def visibility_graph(data, value, window_size=30, step_size=1, batch_size=7):
    frames = []
    vis_frames = []
    targets = []
    values = data.values
    vis_col = data[value].tolist()
    l = len(data)

    for i in range(0, l, step_size):
        end = i + window_size
        if end > l:
            frames.append(values[l - window_size : l])
            targets.append(values[l - 1])
            vis_frames.append(vis_col[l - window_size : l])
            break

        frames.append(values[i:end])
        vis_frames.append(vis_col[i:end])
        targets.append(values[end])

    ll = len(frames)
    graphs = []
    for i in range(ll):
        frame = frames[i]
        vis = vis_frames[i]
        target = targets[i]
        G = nx.visibility_graph(vis)
        temp = from_networkx(G)
        edge_index = temp.edge_index
        x = torch.tensor(frame, dtype=torch.float)
        y = torch.tensor(target, dtype=torch.float)
        graph = Data(x=x, edge_index=edge_index, y=y)
        graphs.append(graph)

    graph_loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

    return graph_loader

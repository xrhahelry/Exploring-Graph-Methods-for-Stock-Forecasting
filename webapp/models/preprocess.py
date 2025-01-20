import networkx as nx
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops
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
            frames.append(values[l - window_size - 1 : l - 1])
            targets.append(values[l - 1])
            vis_frames.append(vis_col[l - window_size - 1 : l - 1])
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

        edge_index, _ = remove_self_loops(edge_index)
        # Add edge weights (absolute difference between node values)
        edge_weight = torch.tensor([abs(vis[u] - vis[v]) for u, v in G.edges() if u != v], dtype=torch.float)
        
        x = torch.tensor(frame, dtype=torch.float)
        y = torch.tensor(target, dtype=torch.float)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
        graphs.append(graph)

    train, val = train_test_split(graphs, test_size=0.2, random_state=12)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def test_visibility_graph(data, value, window_size=30, step_size=1, batch_size=7):
    frames = []
    vis_frames = []
    targets = []
    values = data.values
    vis_col = data[value].tolist()
    l = len(data)

    for i in range(0, l, step_size):
        end = i + window_size
        if end > l:
            frames.append(values[l - window_size - 1 : l - 1])
            targets.append(values[l - 1])
            vis_frames.append(vis_col[l - window_size - 1 : l - 1])
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

        edge_index, _ = remove_self_loops(edge_index)
        edge_weight = torch.tensor([abs(vis[u] - vis[v]) for u, v in G.edges() if u != v], dtype=torch.float)

        x = torch.tensor(frame, dtype=torch.float)
        y = torch.tensor(target, dtype=torch.float)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
        graphs.append(graph)

    return graphs


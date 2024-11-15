import json
import threading

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.utils.convert import from_networkx, to_networkx


def normal_graph(G):
    # G = to_networkx(graph)
    plt.figure(1, figsize=(10, 6))
    nx.draw(G, with_labels=True)


def los_graph(G, tensor):
    plt.figure(2, figsize=(10, 6))
    plt.title("Line-of-Sight Connectivity")
    pos = {x: (x, 0) for x in range(len(tensor))}
    nx.draw_networkx_nodes(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels=None)
    nx.draw_networkx_edges(
        G,
        pos,
        connectionstyle="arc3,rad=-1.57079632679",
        arrows=True,
        arrowstyle="<->",
        arrowsize=10,
    )


def con_graph(G, tensor):
    plt.figure(3, figsize=(10, 6))
    plt.title("Time Series values with Connectivity")
    pos = {i: (i, v) for i, v in enumerate(tensor)}
    nx.draw_networkx_nodes(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels=None)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="<->", arrowsize=10)


df = pd.read_csv("./data/Nabil Bank Limited ( NABIL ).csv")
data = df.head(30)

avg = data["traded_amount"] / data["traded_quantity"]
avg = avg.to_list()

G = nx.visibility_graph(avg)

normal_graph(G)
los_graph(G, avg)
con_graph(G, avg)
plt.show()

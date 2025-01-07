import graph_creation as gc
import numpy as np
import pandas as pd
import torch
from decorators import track_execution
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader


@track_execution
def prepare_stock(df, scaler):
    df = df.drop(columns=["status", "published_date"])
    df["per_change"] = df["per_change"].fillna(0)
    if df.isnull().values.any() or np.isinf(df.values).any():
        df = df.fillna(df.mean())
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.mean())
    cols = df.columns
    data = scaler.fit_transform(df)
    df = pd.DataFrame(data, columns=cols)
    return df


@track_execution
def create_graphs(data, vis_col="close", window_size=30, step_size=20, batch_size=32):
    graphs = gc.create_graphs(data, vis_col="close", window_size=30, step_size=20)
    torch.save(graphs, "./gcn/graphs.pt")

    train, val = train_test_split(graphs, test_size=0.2, random_state=12)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

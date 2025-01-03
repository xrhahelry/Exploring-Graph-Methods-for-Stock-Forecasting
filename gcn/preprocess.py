import graph_creation as gc
import numpy as np
import pandas as pd
from decorators import track_execution
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader


@track_execution
def prepare_stocks(df, start_date, scaler):
    df = df.drop(columns=["status"])
    # df["per_change"] = df["per_change"].fillna(0)
    df["published_date"] = pd.to_datetime(df["published_date"])
    df.set_index("published_date", inplace=True)
    df = df[df.index >= start_date]
    if df.isnull().values.any() or np.isinf(df.values).any():
        df = df.fillna(df.mean())
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.mean())
    og_index = df.index
    cols = df.columns
    data = scaler.fit_transform(df)
    df = pd.DataFrame(data, columns=cols, index=og_index)
    return df


@track_execution
def prepare_stock(df, scaler):
    df = df.drop(columns=["status"])
    # df["per_change"] = df["per_change"].fillna(0)
    df["published_date"] = pd.to_datetime(df["published_date"])
    df.set_index("published_date", inplace=True)
    if df.isnull().values.any() or np.isinf(df.values).any():
        df = df.fillna(df.mean())
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.mean())
    og_index = df.index
    cols = df.columns
    data = scaler.fit_transform(df)
    df = pd.DataFrame(data, columns=cols, index=og_index)
    return df


@track_execution
def create_graphs(
    predictee, stocks, vis_col="close", window_size=30, step_size=20, batch_size=32
):
    graphs = gc.create_graphs(
        predictee, stocks, vis_col="close", window_size=30, step_size=20
    )

    train, val = train_test_split(graphs, test_size=0.2, random_state=12)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

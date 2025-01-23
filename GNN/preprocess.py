import numpy as np
import pandas as pd
from decorators import track_execution
import graph_creation as gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader


def prepare_stock(df, scaler, start_date, end_date):
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
    data = scaler.transform(df)
    df = pd.DataFrame(data, columns=cols, index=og_index)
    df = df[df.index < end_date]
    return df

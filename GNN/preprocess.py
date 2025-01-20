import numpy as np
import pandas as pd
from decorators import track_execution


@track_execution
def prepare_stock(df, scaler):
    df = df.drop(columns=["status", "published_date"])
    df["per_change"] = df["per_change"].fillna(0)
    if df.isnull().values.any() or np.isinf(df.values).any():
        df = df.fillna(df.mean())
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.mean())
    cols = df.columns
    data = scaler.transform(df)
    df = pd.DataFrame(data, columns=cols)
    return df

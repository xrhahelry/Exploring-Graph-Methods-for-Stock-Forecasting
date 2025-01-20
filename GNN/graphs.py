import joblib
import pandas as pd
import graph_creation as gc
import preprocess as pp


window_size = 30
step_size = 3
vis_col = "close"
batch_size = 32
graph_name = "mrg_nib"

scaler = joblib.load("./GNN/scalers/mrg_nib.pkl")
df = pd.read_csv("./data/fundamental data/merged share/NIB.csv")
df = pp.prepare_stock(df, scaler)

gc.create_graphs(df, vis_col, window_size, step_size, graph_name)

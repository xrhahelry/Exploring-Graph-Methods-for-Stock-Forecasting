import json

import nn as NN
import numpy as np
import pandas as pd
import preprocess as pp
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
import joblib

make_new_graph = True
window_size = 30
step_size = 3
vis_col = "close"
batch_size = 32

if make_new_graph:
    scaler = joblib.load("./GNN/scalers/mrg_nib.pkl")
    df = pd.read_csv("./data/fundamental data/merged share/NIB.csv")
    df = pp.prepare_stock(df, scaler)

    train_graphs, val_graphs = pp.create_graphs(
        df, vis_col, window_size, step_size, batch_size
    )
else:
    graphs = torch.load("./GNN/graphs/.pt")
    train, val = train_test_split(graphs, test_size=0.2, random_state=12)
    train_graphs = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_graphs = DataLoader(val, batch_size=batch_size, shuffle=False)

# input_size = 7
# hidden_size = 512
# output_size = 7
# epochs = 100
# learning_rate = 1e-3

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = NN.GCN(input_size, hidden_size, output_size)
# model = model.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# ls_fn = nn.MSELoss()

# model.train()
# for epoch in range(epochs):
#     total_loss = 0
#     for batch in train_graphs:
#         batch = batch.to(device)
#         optimizer.zero_grad()
#         output = model(batch.x, batch.edge_index, batch.batch)
#         output = output.view(-1)
#         loss = ls_fn(output, batch.y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# model.eval()
# predictions = []
# ground_truth = []
# for batch in val_graphs:
#     batch = batch.to(device)
#     with torch.no_grad():
#         output = model(batch.x, batch.edge_index, batch.batch)
#         output = output.view(-1)
#         predictions.append(output.cpu().numpy())
#         ground_truth.append(batch.y.cpu().numpy())

# predictions = np.concatenate(predictions, axis=0)
# ground_truth = np.concatenate(ground_truth, axis=0)

# mse = mean_squared_error(ground_truth, predictions)
# mae = mean_absolute_error(ground_truth, predictions)

# print(f"Mean Squared Error (MSE): {mse:.4f}")
# print(f"Mean Absolute Error (MAE): {mae:.4f}")

import nn as NN
import numpy as np
import pandas as pd
import preprocess as pp
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../nabil.csv")
cols = df.columns
scaler = StandardScaler()
data = scaler.fit_transform(df)
df = pd.DataFrame(data, columns=cols)
train_graphs, test_graphs = pp.visibility_graph(
    data=df, value="open", window_size=30, step_size=20
)

input_size = 7
hidden_size = 64
output_size = 7
epochs = 100
learning_rate = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NN.GCN(input_size, hidden_size, output_size)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
ls_fn = nn.MSELoss()

model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in train_graphs:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch.x, batch.edge_index, batch.batch)
        output = output.view(-1)
        loss = ls_fn(output, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

print(f"Total Loss: {total_loss}")

model.eval()
predictions = []
ground_truth = []
for batch in test_graphs:
    batch = batch.to(device)
    with torch.no_grad():
        output = model(batch.x, batch.edge_index, batch.batch)
        output = output.view(-1)
        predictions.append(output.cpu().numpy())
        ground_truth.append(batch.y.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)
ground_truth = np.concatenate(ground_truth, axis=0)

mse = mean_squared_error(ground_truth, predictions)
mae = mean_absolute_error(ground_truth, predictions)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

predictions = predictions.reshape(-1, 7)
ground_truth = ground_truth.reshape(-1, 7)

predictions = scaler.inverse_transform(predictions)
ground_truth = scaler.inverse_transform(ground_truth)

prd = pd.DataFrame(predictions, columns=cols)
prd.to_csv("../predictions.csv", index=False)
gdt = pd.DataFrame(ground_truth, columns=cols)
gdt.to_csv("../ground_truth.csv", index=False)

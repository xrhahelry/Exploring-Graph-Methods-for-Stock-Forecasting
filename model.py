import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx, to_networkx

df = pd.read_csv("./data/Nabil Bank Limited ( NABIL ).csv")
df = df.drop(columns=["published_date", "status"])

df["per_change"] = df["per_change"].fillna(0)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data)

data = scaled_df[:-1]
target = scaled_df.shift(-1)[:-1]

node_features = torch.tensor(data.values, dtype=torch.float)
target_tensor = torch.tensor(target.values, dtype=torch.float)

num_nodes = len(node_features)

edge_index = torch.tensor(
    [[i, i + 1] for i in range(num_nodes - 1)]
    + [[i + 1, i] for i in range(num_nodes - 1)],
    dtype=torch.long,
).t()

graph = Data(x=node_features, edge_index=edge_index)


class GNNForecastor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GNNForecastor, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)
        return x


input_size = node_features.shape[1]
hidden_size = 16
output_size = input_size
epochs = 100
learning_rate = 1e-2

model = GNNForecastor(input_size, hidden_size, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
ls_fn = nn.MSELoss()

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    output = model(graph.x, graph.edge_index)

    loss = ls_fn(output, target_tensor)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch +1}/{epochs}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    predicted = model(graph.x, graph.edge_index)

predicted_og = scaler.inverse_transform(predicted.numpy())
target_og = scaler.inverse_transform(target_tensor.numpy())

predicted_og_round = np.round(predicted_og, 2)
target_og_round = np.round(target_og, 2)

np.set_printoptions(precision=2, suppress=True, linewidth=100)
for i in range(num_nodes):
    print(f"Predicted: {predicted_og_round[i]}")
    print(f"   Actual: {target_og_round[i]}")
    print("")

pog = torch.tensor(predicted_og, dtype=torch.float)
tog = torch.tensor(target_og, dtype=torch.float)
print(pog)
print(tog)

sum_pog = torch.sum(pog, dim=1)
sum_tog = torch.sum(tog, dim=1)
print(sum_pog)
print(sum_tog)


mse = F.mse_loss(sum_pog, sum_tog)
print(f"Mean Squared Error: {mse.item():.4f}")

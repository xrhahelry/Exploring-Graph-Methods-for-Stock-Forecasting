import torch
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.io as pio

# Load the trained model
def load_model(bank: str, model_type: str, input_size: int, hidden_size: int, output_size: int):
    model_path = f"./GNN/models/{bank}_{model_type}.pth"
    if model_type == 'GCN':
        model = NN.GCN(input_size, hidden_size, output_size)
    elif model_type == 'GAT':
        model = NN.GAT(input_size, hidden_size, output_size)
    else:
        raise ValueError("Invalid model type specified")
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Generate predictions
def generate_predictions(model, graphs, scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []
    ground_truth = []
    for graph in graphs:
        graph = graph.to(device)
        with torch.no_grad():
            if isinstance(model, NN.GCN):
                output = model(graph.x, graph.edge_index, torch.zeros(graph.x.size(0), dtype=torch.long))
            elif isinstance(model, NN.GAT):
                output = model(graph.x, graph.edge_index, graph.edge_weight, torch.zeros(graph.x.size(0), dtype=torch.long))
            else:
                raise ValueError("Invalid model type specified")
            predictions.append(output.cpu().numpy())
            ground_truth.append(graph.y.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    
    # Reshape to match the original feature dimensions
    predictions = predictions.reshape(-1, 7)
    ground_truth = ground_truth.reshape(-1, 7)
    
    # Inverse transform to the original scale
    predictions = scaler.inverse_transform(predictions)
    ground_truth = scaler.inverse_transform(ground_truth)
    
    return predictions, ground_truth

def visualize_results(predictions, ground_truth):
    graphs = []
    feature_columns = ['open', 'high', 'low', 'close', 'per_change', 'traded_quantity', 'traded_amount']
    for i, col in enumerate(feature_columns):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(ground_truth))),
            y=ground_truth[:, i],
            mode='lines+markers',
            name='Ground Truth',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(predictions))),
            y=predictions[:, i],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='orange')
        ))
        fig.update_layout(
            title=f"Feature {col} (Ground Truth vs Predicted)",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_white"
        )
        graphs.append(pio.to_html(fig, full_html=False))
    return graphs

def predict_main(bank: str, model_type: str):
    # Load the scaler
    scaler = joblib.load(f"./GNN/scalers/{bank}.pkl")
    
    # Load the preprocessed graphs
    graphs = torch.load(f"./GNN/graphs/{bank}_graphs.pt")
    
    # Load the model
    input_size = 7  # Number of features
    hidden_size = 64
    output_size = 7
    model = load_model(bank, model_type, input_size, hidden_size, output_size)
    
    # Generate predictions
    predictions, ground_truth = generate_predictions(model, graphs, scaler)
    
    # Visualize results
    graphs_html = visualize_results(predictions, ground_truth)
    return graphs_html

    
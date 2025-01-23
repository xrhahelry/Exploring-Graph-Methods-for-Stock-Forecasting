import torch
import numpy as np
import pandas as pd
import models.nn as NN
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Load the scaler
scaler = joblib.load('../GNN/scalers/hyd_bpcl.pkl')

# Load the trained model
def load_model(model_path, input_size, hidden_size, output_size, model_type):
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
def generate_predictions(model, graphs, model_type, df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    predictions = []
    ground_truth = []
    for batch in graphs:
        batch = batch.to(device)
        with torch.no_grad():
            if model_type == 'GCN':
                output = model(batch.x, batch.edge_index, batch.batch)
            elif model_type == 'GAT':
                output = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
            else:
                raise ValueError("Invalid model type specified")
            
            output = output.view(-1)
            predictions.append(output.cpu().numpy())
            ground_truth.append(batch.y.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    
    # Reshape to match the original feature dimensions
    predictions = predictions.reshape(-1, 7)
    ground_truth = ground_truth.reshape(-1, 7)
    
    # Inverse transform to the original scale
    predictions_original = scaler.inverse_transform(predictions)
    ground_truth_original = scaler.inverse_transform(ground_truth)
    
    # Convert to DataFrames
    predictions_df = pd.DataFrame(predictions_original, columns=df.columns)
    ground_truth_df = pd.DataFrame(ground_truth_original, columns=df.columns)
    
    # Add a 'published_date' column for plotting
    predictions_df['published_date'] = np.arange(len(predictions_df))
    ground_truth_df['published_date'] = np.arange(len(ground_truth_df))
    
    return predictions_df, ground_truth_df

def visualize_results(predictions_df, ground_truth_df):
    # Create a list to store all the graphs
    graphs_html = []
    
    # Iterate over each feature (column) except 'published_date'
    for feature in predictions_df.columns:
        if feature != 'published_date':
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ground_truth_df['published_date'], 
                y=ground_truth_df[feature], 
                mode='lines+markers', 
                name='Ground Truth', 
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=predictions_df['published_date'], 
                y=predictions_df[feature], 
                mode='lines+markers', 
                name='Predicted', 
                line=dict(color='orange')
            ))
            fig.update_layout(
                title=f"Test Set: <b>{feature}</b> (Ground Truth vs Predicted)",
                xaxis_title="Published Date",
                yaxis_title=feature,
                template="plotly_white"
            )
            
            # Convert the figure to HTML and add it to the list
            graph_html = pio.to_html(fig, full_html=False)
            graphs_html.append(graph_html)
    
    return graphs_html



def main(model_type):
    #load the preprocessed graphs
    test_graphs_path = "../GNN/graphs/hyd_bpcl_test.pt"
    test_graphs = torch.load(test_graphs_path)

    # Load the data
    data_path = "../data/fundamental data/hydro/BPCL.csv"
    df = pd.read_csv(data_path)
    df = df.drop(columns=['published_date', 'status'])
    df['per_change'] = df["per_change"].fillna(0)

    #load the model
    input_size = 7  
    hidden_size = 512
    output_size = 7 

    model_path = "../GNN/models/gcn_bpcl.pth" if model_type == 'GCN' else '../GNN/models/gat.bpcl.pth'
    model = load_model(model_path, input_size, hidden_size, output_size, model_type)

    # Generate predictions for the test set
    test_predictions_df, test_ground_truth_df = generate_predictions(model, test_graphs, model_type, df)

    # Visualize the test set results
    graph_html = visualize_results(test_predictions_df, test_ground_truth_df)
    return graph_html

if __name__ == "__main__":
    graph_html = main('GCN')  # Default to GCN if run directly
    for graph in graphs.html:
        print(graph)

import torch
import numpy as np
import pandas as pd
import models.preprocess as pp
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
scaler = joblib.load('models/scaler2.pkl')

# Load the trained model
def load_model(model_path, input_size, hidden_size, output_size):
    model = NN.GAT(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Generate predictions
def generate_predictions(model, graphs, df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []
    ground_truth = []
    for batch in graphs:
        batch = batch.to(device)
        with torch.no_grad():
            output = model(batch.x, batch.edge_index,batch.edge_weight, batch.batch)
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

# Visualize the results
def visualize_results(predictions_df, ground_truth_df, feature_columns):
    graphs = []
    for col in feature_columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ground_truth_df['published_date'], 
            y=ground_truth_df[col], 
            mode='lines+markers', 
            name='Ground Truth', 
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=predictions_df['published_date'], 
            y=predictions_df[col], 
            mode='lines+markers', 
            name='Predicted', 
            line=dict(color='orange')
        ))
        fig.update_layout(
            title=f"Test Set: <b>{col}</b> (Ground Truth vs Predicted)",
            xaxis_title="Published Date",
            yaxis_title=col,
            template="plotly_white"
        )
        
        # Convert the figure to HTML
        graph_html = pio.to_html(fig, full_html=False)
        graphs.append(graph_html)
    
    return graphs

def main():
    # Load the data
    data_path = "../data/fundamental data/commercial bank/ADBL.csv"
    df = pd.read_csv(data_path)
    df = df.drop(columns=['published_date', 'status'])
    df['per_change'] = df["per_change"].fillna(0)

    # Train-test split
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:]

    # Normalize the data
    train_data = scaler.transform(train_df)
    test_data = scaler.transform(test_df)

    # Convert back to DataFrame
    train_df = pd.DataFrame(train_data, columns=df.columns)
    test_df = pd.DataFrame(test_data, columns=df.columns)

    # Generate graphs for training, validation, and testing
    train_graphs, val_graphs = pp.visibility_graph(
        data=train_df, value="open", window_size=30, step_size=20
    )

    test_graphs = pp.test_visibility_graph(
        data=test_df, value="open", window_size=30, step_size=20
    )

    # Load the model
    input_size = 7  # Number of features
    hidden_size = 64
    output_size = 7  # Number of features

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("models/gat(adbl).pth", input_size, hidden_size, output_size)
    model = model.to(device)
    model.eval()

    # Generate predictions for the test set
    test_predictions_df, test_ground_truth_df = generate_predictions(model, test_graphs, df)

    # Feature columns
    feature_columns = ['open', 'high', 'low', 'close', 'per_change', 'traded_quantity', 'traded_amount']

    # Visualize the test set results
    test_ground_truth_df = test_ground_truth_df[:-1]
    graphs = visualize_results(test_predictions_df, test_ground_truth_df, feature_columns)
    return graphs

if __name__ == "__main__":
    main()
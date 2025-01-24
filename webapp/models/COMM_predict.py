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

cols = ["ADBL", "CZBIL", "EBL", "GBIME", "HBL", "KBL", "MBL", "NABIL", "NBL", "NICA", "NMB", "PCBL", "PRVU", "SANIMA", "SBI", "SBL", "SCB"]
#load the scaler
scaler = joblib.load('../GNN/scalers/comm.pkl')

#Load the trained model
def load_model(model_path, input_size, hidden_size, output_size):
    model = NN.GCN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def split_predictions(predictions, cols):
    predictions = np.array(predictions)

    dataframes = []

    for i in range(len(cols)):
        df = pd.DataFrame({'Close': predictions[:, i]})

        for j in range(6):
            df[f'col_{j+1}'] = 0

        dataframes.append(df)

    return dataframes

def generate_predictions(model, graphs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    predictions = []
    ground_truth = []
    for batch in graphs:
        batch = batch.to(device)
        with torch.no_grad():
            output = model(batch.x, batch.edge_index, batch.batch)
            output = output.view(-1)
            predictions.append(output.cpu().numpy())
            ground_truth.append(batch.y.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)

    # Reshape to match the original feature dimensions
    predictions = predictions.reshape(-1, 15)
    ground_truth = ground_truth.reshape(-1, 15)

    predictions = split_predictions(predictions, cols)
    ground_truth = split_predictions(ground_truth, cols)

    og_cols = predictions[0].columns
    # Inverse transform to the original scale
    predictions_original = [scaler.inverse_transform(prediction) for prediction in predictions]
    ground_truth_original = [scaler.inverse_transform(truth) for truth in ground_truth]

    predictions_original = [pd.DataFrame(df, columns=og_cols) for df in predictions_original]
    ground_truth_original = [pd.DataFrame(df, columns=og_cols) for df in ground_truth_original]

    # Add a 'published_date' column for plotting
    for df in predictions_original:
        df['published_date'] = np.arange(len(df))

    for df in ground_truth_original:
        df['published_date'] = np.arange(len(df))

    return predictions_original, ground_truth_original


# Function to visualize results
def visualize_results(predictions_original, ground_truth_original):
    graphs_html = []
    
    for i, col in enumerate(cols):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ground_truth_original[i]['published_date'][:-2], 
            y=ground_truth_original[i]['Close'][:-2], 
            mode='lines+markers', 
            name='Ground Truth', 
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=predictions_original[i]['published_date'], 
            y=predictions_original[i]['Close'], 
            mode='lines+markers', 
            name='Predicted', 
            line=dict(color='orange')
        ))
        fig.update_layout(
            title=f"Test Set: <b>{col}</b> (Ground Truth vs Predicted)",
            xaxis_title="Published Date",
            yaxis_title="Close Price",
            template="plotly_white"
        )
        
        # Convert the figure to HTML and add it to the list
        graph_html = pio.to_html(fig, full_html=False)
        graphs_html.append(graph_html)
    
    return graphs_html



def main():
    # Load the preprocessed graphs
    graph = torch.load("../GNN/graphs/test_graphs/comm_test.pt", weights_only=False)
    test_graphs = DataLoader(graph, batch_size=32, shuffle=False)

    # Load the model
    input_size = 7  
    hidden_size = 128
    output_size = 17

    model_path = "../GNN/models/gcn_comm2.pth" 
    model = load_model(model_path, input_size, hidden_size, output_size)

    # Generate predictions for the test set
    test_predictions_df, test_ground_truth_df = generate_predictions(model, test_graphs)

    # Visualize the test set results
    graphs_html = visualize_results(test_predictions_df, test_ground_truth_df)
    return graphs_html

if __name__ == "__main__":
    graph_html = main()  
    for graph in graphs.html:
        print(graph)

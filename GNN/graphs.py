import json
import graph_creation as gc
import joblib
import pandas as pd
import preprocess as pp


def single():
    window_size = 30
    step_size = 3
    vis_col = "close"
    graph_name = "mrg_nib_wt"

    scaler = joblib.load("./scalers/mrg_nib.pkl")
    df = pd.read_csv("../data/fundamental data/merged share/NIB.csv")
    df = pp.prepare_stock(df, scaler)

    gc.create_graphs_singular(
        df, vis_col, window_size, step_size, graph_name=graph_name
    )


def sector():
    window_size = 30
    step_size = 3
    vis_col = "close"
    graph_name = "hyd"

    with open("./GNN/stocks.json") as json_file:
        stock_paths = json.load(json_file)
        predict_path = stock_paths["hydro_predict"]
        other_paths = stock_paths["hydro"]

    predictee = pd.read_csv(f"./data/{predict_path}")
    stocks = [pd.read_csv(f"./data/{path}") for path in other_paths]

    scaler = joblib.load(f"./GNN/scalers/{graph_name}.pkl")
    graph_name = graph_name + "_test"
    start_date = "2023-08-15"
    end_date = "2024-12-30"
    predictee = pp.prepare_stock(predictee, scaler, start_date, end_date)
    stocks = [pp.prepare_stock(stock, scaler, start_date, end_date) for stock in stocks]

    gc.create_graphs(
        predictee,
        stocks,
        vis_col=vis_col,
        window_size=window_size,
        step_size=step_size,
        graph_name=graph_name,
    )


sector()

import graph_creation as gc
import joblib
import pandas as pd
import preprocess as pp


def single():
    window_size = 30
    step_size = 3
    vis_col = "close"
    batch_size = 32
    graph_name = "mrg_nib_wt"

    scaler = joblib.load("./scalers/mrg_nib.pkl")
    df = pd.read_csv("../data/fundamental data/merged share/NIB.csv")
    df = pp.prepare_stock(df, scaler)

    gc.create_graphs_singular(df, vis_col, window_size, step_size, graph_name=graph_name)

def sector():
    make_new_graph = True
    window_size = 30
    step_size = 3
    vis_col = "close"
    graph_name = "comm_bank"

    with open("./GNN/stocks.json") as json_file:
        stock_paths = json.load(json_file)
        predict_path = stock_paths["predict"]
        other_paths = stock_paths["other"]

    predictee = pd.read_csv(f"./data/{predict_path}")
    stocks = [pd.read_csv(f"./data/{path}") for path in other_paths]

    scaler = joblib.load(f"./GNN/scalers/{graph_name}.pkl")

    predictee = pp.prepare_stock(predictee, scaler)
    start_date = predictee.index[0]
    stocks = [pp.prepare_stocks(stock, start_date, scaler) for stock in stocks]

    gc.create_graphs(
        predictee,
        stocks,
        graph_name,
        vis_col=vis_col,
        window_size=window_size,
        step_size=step_size,
        
    )

if __name__=="__main__":
    single()
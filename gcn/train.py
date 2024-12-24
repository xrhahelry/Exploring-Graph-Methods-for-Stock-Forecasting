import pandas as pd
import preprocess as pp

df = pd.read_csv("./data/nabil.csv")

graphs = pp.visibility_graph(data=df, value="open", window_size=30, step_size=20)

print(graphs)

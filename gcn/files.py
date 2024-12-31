import json
import os


def get_file_names():
    sectors = {}
    for sector in os.listdir("./data/fundamental data/"):
        sectors[sector] = [
            name.replace(".csv", "")
            for name in os.listdir(f"./data/fundamental data/{sector}")
        ]

    with open("./visibility_graphs/fundamental_data_graph/data.json", "w") as json_file:
        json.dump(sectors, json_file, indent=4)


def get_stock_names():
    with open("./graph_construction/files.json", "r") as json_file:
        files = json.load(json_file)

    stocks = []
    for sector in files:
        for institute in files[sector]:
            stocks.append(institute)

    data = {"stocks": stocks}

    with open("./graph_construction/stocks.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


get_file_names()

import json
import os


def get_file_names():
    sectors = {}
    for sector in os.listdir("./data/fundamental data/"):
        sectors[sector] = [
            name.replace(".csv", "")
            for name in os.listdir(f"./data/fundamental data/{sector}")
        ]

    with open("./gcn/data.json", "w") as json_file:
        json.dump(sectors, json_file, indent=4)


def get_stock_names():
    with open("./gcn/data.json", "r") as json_file:
        files = json.load(json_file)

    stocks = []
    for sector in files:
        for institute in files[sector]:
            stocks.append(f"fundamental data/{sector}/{institute}.csv")

    data = {"other": stocks}

    with open("./gcn/stocks.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


get_file_names()
get_stock_names()

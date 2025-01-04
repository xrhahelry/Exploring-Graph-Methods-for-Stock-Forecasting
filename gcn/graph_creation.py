import networkx as nx
import pandas as pd
import torch
from decorators import track_execution
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx


@track_execution
def create_graphs(predictee, stocks, vis_col="close", window_size=30, step_size=20):
    l = len(predictee)
    predict_frames = []
    predict_dates = predictee.index
    predict_values = predictee["close"].values
    stocks_frames = []
    targets = []

    for i in range(0, l, step_size):
        frames = []
        end = i + window_size
        if end > l:
            predict_frames.append(predictee[l - window_size - 1 : l - 1])
            targets.append(predict_values[l - 1])

            start_date = predict_dates[l - window_size - 1]
            end_date = predict_dates[l - 2]

            for stock in stocks:
                frame = stock[stock.index >= start_date]
                frame = frame[frame.index <= end_date]
                frames.append(frame)

            stocks_frames.append(frames)
            break

        predict_frames.append(predictee[i:end])
        targets.append(predict_values[end])

        start_date = predict_dates[i]
        end_date = predict_dates[end - 1]

        for stock in stocks:
            frame = stock[stock.index >= start_date]
            frame = frame[frame.index <= end_date]
            frames.append(frame)

        stocks_frames.append(frames)

    predict_edge_indexes = [
        from_networkx(nx.visibility_graph(frame[vis_col])).edge_index.to(torch.int64)
        for frame in predict_frames
    ]
    predict_frame_dates = [
        frame.index.strftime("%Y%m%d").astype(int).tolist() for frame in predict_frames
    ]
    stocks_edge_indexes = [
        [
            from_networkx(nx.visibility_graph(frame[vis_col])).edge_index.to(
                torch.int64
            )
            for frame in frames
        ]
        for frames in stocks_frames
    ]
    stocks_frames_dates = [
        [frame.index.strftime("%Y%m%d").astype(int).tolist() for frame in frames]
        for frames in stocks_frames
    ]

    graphs = []
    for i in range(len(predict_frames)):
        predict_x = torch.tensor(predict_frames[i].values, dtype=torch.float)
        predict_dates = torch.tensor(predict_frame_dates[i], dtype=torch.int)
        predict_edge_index = predict_edge_indexes[i]
        predict_graph = Data(
            x=predict_x, edge_index=predict_edge_index, dates=predict_dates
        )

        main_x = predict_x
        main_edge_index = predict_edge_index
        main_y = torch.tensor(targets[i], dtype=torch.float)
        offset = predict_graph.x.size(0)

        for j in range(len(stocks_frames[i])):
            stock_x = torch.tensor(stocks_frames[i][j].values, dtype=torch.float)
            stock_dates = torch.tensor(stocks_frames_dates[i][j], dtype=torch.int)
            stock_edge_index = stocks_edge_indexes[i][j]
            stock_graph = Data(
                x=stock_x, edge_index=stock_edge_index, dates=stock_dates
            )

            common_dates = torch.tensor(
                [date for date in predict_dates if date in stock_dates],
                dtype=torch.int,
            )

            new_edge_index = []
            for date in common_dates:
                nodes_in_predict = (predict_graph.dates == date).nonzero(as_tuple=True)[
                    0
                ]
                nodes_in_stock = (stock_graph.dates == date).nonzero(as_tuple=True)[0]

                for node1 in nodes_in_predict:
                    for node2 in nodes_in_stock:
                        new_edge_index.append([node1.item(), node2.item() + offset])

            new_edge_index = (
                torch.tensor(new_edge_index, dtype=torch.int).t().contiguous()
            )
            main_x = torch.cat([main_x, stock_graph.x], dim=0)

            main_edge_index = torch.cat(
                [main_edge_index, stock_graph.edge_index + offset, new_edge_index],
                dim=1,
            )
            offset += stock_graph.x.size(0)

        graphs.append(Data(x=main_x, edge_index=main_edge_index, y=main_y))

    torch.save(graphs, "./gcn/graphs.pt")
    return graphs

import networkx as nx
import pandas as pd
import torch
from decorators import track_execution
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils.convert import from_networkx


@track_execution
def create_graphs_singular(
    data, vis_col, window_size=30, step_size=1, batch_size=7, graph_name="graphs.pt"
):
    frames = []
    vis_frames = []
    targets = []
    values = data.values
    vis_col = data[vis_col].tolist()
    l = len(data)

    for i in range(0, l, step_size):
        end = i + window_size
        if end >= l:
            frames.append(values[l - window_size - 1 : l - 1])
            targets.append(values[l - 1])
            vis_frames.append(vis_col[l - window_size - 1 : l - 1])
            break

        frames.append(values[i:end])
        vis_frames.append(vis_col[i:end])
        targets.append(values[end])

    ll = len(frames)
    graphs = []
    for i in range(ll):
        frame = frames[i]
        vis = vis_frames[i]
        target = targets[i]
        G = nx.visibility_graph(vis)
        temp = from_networkx(G)
        edge_index = temp.edge_index

        x = torch.tensor(frame, dtype=torch.float)
        y = torch.tensor(target, dtype=torch.float)
        graph = Data(x=x, edge_index=edge_index, y=y)
        graphs.append(graph)

    torch.save(graphs, f"./graphs/{graph_name}.pt")
    return graphs


@track_execution
def create_graphs_singular_with_edge_weights(
    data, vis_col, window_size=30, step_size=1, batch_size=7, graph_name="graphs.pt"
):
    frames = []
    vis_frames = []
    targets = []
    values = data.values
    vis_col = data[vis_col].tolist()
    l = len(data)

    for i in range(0, l, step_size):
        end = i + window_size
        if end >= l:
            frames.append(values[l - window_size - 1 : l - 1])
            targets.append(values[l - 1])
            vis_frames.append(vis_col[l - window_size - 1 : l - 1])
            break

        frames.append(values[i:end])
        vis_frames.append(vis_col[i:end])
        targets.append(values[end])

    ll = len(frames)
    graphs = []
    for i in range(ll):
        frame = frames[i]
        vis = vis_frames[i]
        target = targets[i]
        G = nx.visibility_graph(vis)
        temp = from_networkx(G)
        edge_index = temp.edge_index

        edge_index, _ = remove_self_loops(edge_index)
        # Fix: Assign edge weights only for unique edges
        edge_weight = torch.tensor(
            [abs(vis[u] - vis[v]) for u, v in zip(edge_index[0], edge_index[1])],
            dtype=torch.float,
        )
        print(edge_index.shape)
        print(edge_weight.shape)
        x = torch.tensor(frame, dtype=torch.float)
        y = torch.tensor(target, dtype=torch.float)
        graph = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
        graphs.append(graph)

    torch.save(graphs, f"./graphs/{graph_name}.pt")
    return graphs


@track_execution
def create_graphs(
    predictee,
    stocks,
    vis_col="close",
    window_size=30,
    step_size=20,
    graph_name="graphs",
):
    l = len(predictee)
    predict_frames = []
    predict_dates = predictee.index
    stocks_frames = []
    targets = []

    for i in range(0, l, step_size):
        predict_values = []
        frames = []
        end = i + window_size
        if end >= l:
            temp_frame = predictee[l - window_size - 1 : l - 1]
            predict_frames.append(temp_frame)
            t = predictee.iloc[l - 1]["close"]
            predict_values.append(t)
            start_date = predict_dates[l - window_size - 1]
            end_date = predict_dates[l - 2]

            for stock in stocks:
                frame = stock[stock.index >= start_date]
                frame = frame[frame.index <= end_date]

                temp = stock[stock.index > end_date]
                temp = temp[temp.index <= end_date + pd.DateOffset(days=30)]

                if not temp["close"].empty:
                    t = temp["close"].to_list()
                    predict_values.append(t[0])
                else:
                    predict_values.append(0)
                frames.append(frame)

            stocks_frames.append(frames)
            targets.append(predict_values)
            break

        temp_frame = predictee[i:end]
        predict_frames.append(temp_frame)
        t = predictee.iloc[end]["close"]
        predict_values.append(t)

        start_date = predict_dates[i]
        end_date = predict_dates[end - 1]

        for stock in stocks:
            frame = stock[stock.index >= start_date]
            frame = frame[frame.index <= end_date]

            temp = stock[stock.index > end_date]
            temp = temp[temp.index <= end_date + pd.DateOffset(days=30)]

            if not temp["close"].empty:
                t = temp["close"].to_list()
                predict_values.append(t[0])
            else:
                predict_values.append(0)
            frames.append(frame)

        stocks_frames.append(frames)
        targets.append(predict_values)

    predict_edge_indexes = [
        from_networkx(nx.visibility_graph(frame[vis_col])).edge_index.to(torch.int64)
        for frame in predict_frames
    ]
    predict_frame_dates = [
        frame.index.strftime("%Y%m%d").astype(int).tolist() for frame in predict_frames
    ]
    stocks_edge_indexes = [
        [
            from_networkx(nx.visibility_graph(frame[vis_col])).edge_index.to(torch.int)
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
        main_x = torch.tensor(predict_frames[i].values, dtype=torch.float)
        main_dates = torch.tensor(predict_frame_dates[i], dtype=torch.int)
        main_edge_index = predict_edge_indexes[i]
        main_y = torch.tensor(targets[i], dtype=torch.float)
        main_graph = Data(
            x=main_x, edge_index=main_edge_index, y=main_y, dates=main_dates
        )
        offset = main_graph.x.size(0)

        for j in range(len(stocks_frames[i])):
            stock_x = torch.tensor(stocks_frames[i][j].values, dtype=torch.float)
            stock_dates = torch.tensor(stocks_frames_dates[i][j], dtype=torch.int)
            stock_edge_index = stocks_edge_indexes[i][j]
            stock_graph = Data(
                x=stock_x, edge_index=stock_edge_index, dates=stock_dates
            )

            common_dates_mask = torch.isin(main_dates, stock_dates)
            common_dates = main_dates[common_dates_mask]

            new_edge_index = []
            for date in common_dates:
                nodes_in_main = (main_graph.dates == date).nonzero(as_tuple=True)[0]
                nodes_in_stock = (stock_graph.dates == date).nonzero(as_tuple=True)[0]

                for node1 in nodes_in_main:
                    for node2 in nodes_in_stock:
                        new_edge_index.append([node1.item(), node2.item() + offset])
                        new_edge_index.append([node2.item() + offset, node1.item()])

            new_edge_index = (
                torch.tensor(new_edge_index, dtype=torch.int).t().contiguous()
            )
            main_x = torch.cat([main_graph.x, stock_graph.x], dim=0)
            main_dates = torch.cat([main_graph.dates, stock_graph.dates], dim=0)

            main_edge_index = torch.cat(
                [
                    main_graph.edge_index,
                    stock_graph.edge_index + offset,
                    new_edge_index,
                ],
                dim=1,
            )

            offset += stock_graph.x.size(0)

            main_graph = Data(
                x=main_x, edge_index=main_edge_index, y=main_y, dates=main_dates
            )

        graphs.append(main_graph)

    torch.save(graphs, f"./GNN/graphs/{graph_name}.pt")
    return graphs


@track_execution
def create_graphs_with_edge_weights(
    predictee,
    stocks,
    vis_col="close",
    window_size=30,
    step_size=20,
    graph_name="graphs.pt",
):
    l = len(predictee)
    predict_frames = []
    predict_dates = predictee.index
    stocks_frames = []
    targets = []

    for i in range(0, l, step_size):
        predict_values = []
        frames = []
        end = i + window_size
        if end >= l:
            temp_frame = predictee[l - window_size - 1 : l - 1]
            predict_frames.append(temp_frame)
            t = predictee.iloc[l - 1]["close"]
            predict_values.append(t)
            start_date = predict_dates[l - window_size - 1]
            end_date = predict_dates[l - 2]

            for stock in stocks:
                frame = stock[stock.index >= start_date]
                frame = frame[frame.index <= end_date]

                temp = stock[stock.index > end_date]
                temp = temp[temp.index <= end_date + pd.DateOffset(days=30)]

                if not temp["close"].empty:
                    t = temp["close"].to_list()
                    predict_values.append(t[0])
                else:
                    predict_values.append(0)
                frames.append(frame)

            stocks_frames.append(frames)
            targets.append(predict_values)
            break

        temp_frame = predictee[i:end]
        predict_frames.append(temp_frame)
        t = predictee.iloc[end]["close"]
        predict_values.append(t)

        start_date = predict_dates[i]
        end_date = predict_dates[end - 1]

        for stock in stocks:
            frame = stock[stock.index >= start_date]
            frame = frame[frame.index <= end_date]

            temp = stock[stock.index > end_date]
            temp = temp[temp.index <= end_date + pd.DateOffset(days=30)]

            if not temp["close"].empty:
                t = temp["close"].to_list()
                predict_values.append(t[0])
            else:
                predict_values.append(0)
            frames.append(frame)

        stocks_frames.append(frames)
        targets.append(predict_values)

    predict_edge_indexes = [
        from_networkx(nx.visibility_graph(frame[vis_col])).edge_index.to(torch.int64)
        for frame in predict_frames
    ]
    predict_vis_col = [frame[vis_col].to_list() for frame in predict_frames]
    predict_edge_weights = []

    for i, edge_index in enumerate(predict_edge_indexes):
        temp_edge_weight = [
            abs(predict_vis_col[i][u] - predict_vis_col[i][v])
            for u, v in zip(edge_index[0], edge_index[1])
        ]
        predict_edge_weights.append(temp_edge_weight)

    predict_frame_dates = [
        frame.index.strftime("%Y%m%d").astype(int).tolist() for frame in predict_frames
    ]
    stocks_edge_indexes = [
        [
            from_networkx(nx.visibility_graph(frame[vis_col])).edge_index.to(torch.int)
            for frame in frames
        ]
        for frames in stocks_frames
    ]
    stocks_vis_cols = [
        [frame[vis_col].to_list() for frame in frames] for frames in stocks_frames
    ]
    stocks_edge_weights = []

    for i, edge_indexes in enumerate(stocks_edge_indexes):
        temp = []
        for j, edge_index in enumerate(edge_indexes):
            temp_edge_weight = [
                abs(stocks_vis_cols[i][j][u] - stocks_vis_cols[i][j][v])
                for u, v in zip(edge_index[0], edge_index[1])
            ]
            temp.append(temp_edge_weight)
        stocks_edge_weights.append(temp)

    stocks_frames_dates = [
        [frame.index.strftime("%Y%m%d").astype(int).tolist() for frame in frames]
        for frames in stocks_frames
    ]

    graphs = []
    for i in range(len(predict_frames)):
        main_x = torch.tensor(predict_frames[i].values, dtype=torch.float)
        main_dates = torch.tensor(predict_frame_dates[i], dtype=torch.int)
        main_edge_index = predict_edge_indexes[i]
        main_edge_weights = torch.tensor(predict_edge_weights[i], dtype=torch.int)
        main_y = torch.tensor(targets[i], dtype=torch.float)
        main_graph = Data(
            x=main_x,
            edge_index=main_edge_index,
            edge_weight=main_edge_weights,
            y=main_y,
            dates=main_dates,
        )
        offset = main_graph.x.size(0)
        offset_edge = main_graph.edge_weight.size(0)

        for j in range(len(stocks_frames[i])):
            stock_x = torch.tensor(stocks_frames[i][j].values, dtype=torch.float)
            stock_dates = torch.tensor(stocks_frames_dates[i][j], dtype=torch.int)
            stock_edge_index = stocks_edge_indexes[i][j]
            stock_edge_weight = stocks_edge_weights[i][j]
            stock_graph = Data(
                x=stock_x,
                edge_index=stock_edge_index,
                edge_weight=stock_edge_weight,
                dates=stock_dates,
            )

            common_dates_mask = torch.isin(main_dates, stock_dates)
            common_dates = main_dates[common_dates_mask]

            new_edge_index = []
            for date in common_dates:
                nodes_in_main = (main_graph.dates == date).nonzero(as_tuple=True)[0]
                nodes_in_stock = (stock_graph.dates == date).nonzero(as_tuple=True)[0]

                for node1 in nodes_in_main:
                    for node2 in nodes_in_stock:
                        new_edge_index.append([node1.item(), node2.item() + offset])
                        new_edge_index.append([node2.item() + offset, node1.item()])

            new_edge_index = (
                torch.tensor(new_edge_index, dtype=torch.int).t().contiguous()
            )
            new_edge_weights = torch.zeros(new_edge_index.shape[1], dtype=torch.int)
            main_x = torch.cat([main_graph.x, stock_graph.x], dim=0)
            main_dates = torch.cat([main_graph.dates, stock_graph.dates], dim=0)

            main_edge_index = torch.cat(
                [
                    main_graph.edge_index,
                    stock_graph.edge_index + offset,
                    new_edge_index,
                ],
                dim=1,
            )
            main_edge_weight = torch.cat(
                [main_graph.edge_weight, stock_graph.edge_weight, new_edge_weights]
            )

            offset += stock_graph.x.size(0)

            main_graph = Data(
                x=main_x,
                edge_index=main_edge_index,
                edge_weights=main_edge_weight,
                y=main_y,
                dates=main_dates,
            )

        graphs.append(main_graph)

    torch.save(graphs, f"./GNN/graphs/{graph_name}.pt")
    return graphs

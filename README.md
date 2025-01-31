# Exploring Graph Methods for Stock Forecasting
For the longest of time LSTMs and transformers have been the industry standard for time series analysis and anything that involves data as a series.
This project is trying to expand the horizon for time series analysis by employing graph theory for stock forecasting.

## How is graph helpful?
Graph is a mathematical data structure that exists in a non-euclidean space meaning its orientation does not matter.
This begs the question than how will it help in time series where the data is a sequence and the importance of orientation is paramount.

**The Visibility Graph** is what enables graphs to be used for time series analysis. Visibility Graph in action.

![](https://i.imgur.com/CvNlFdu.png)

## Graph Construction
Graph is constructed using [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html), the GNN is also done in [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html). After creation the graph is visualized using [NetworkX](https://networkx.org/documentation/stable/index.html).
## Example graph
![](https://i.imgur.com/XnUgUrH.png)

Graphs will be created using the fundamental data which has features [open, close, high, low, per change, traded amount, traded quantity, published date]

### Nodes
For the graph every node will represent a unique day in the time series data. Every node will have 7 features namely [open, close, high, low, per change, traded amount, traded quantity] now there is one feature remaining that is the published date this does not play any active role in the stock price prediction but this will also be included in every node as a non-training feature. Non-training features in the context of GNNs are features that can be used to uniquely identify a node but will not be used in any way during the training process. The reason for including published date in every node is that later on it will be used when creating inter stock edges.
### Stock Graph
Stock graphs are graphs created using only a single stock's data. In this graph we treat each stock as an independent entity and that is not affected by the other stocks whatsoever. The edges in this graph are created according to the visibility algorithm and convert a single variable time series data into a graph. The main principle behind the visibility algorithm is that for two arbitrary data values ($t_a$, $y_a$) and ($t_b$, $y_b$) will have visibility, and consequently will become two connected nodes of the associated graph, if any other data ($t_c$, $y_c$) placed between them fulfills:
$$y_c<y_b+(y_a-y_b)\frac{t_b-t_c}{t_b-t_a}$$
Now, selecting the feature to be used in the visibility algorithm. We have 7 features available among which open, close, high and low are highly correlated so choosing anyone among them will yield identical results. Traded quantity and amount are not that strong predictor of price so we won’t use them and per change’s value oscillates very heavy which results in a very dense graph which consequently results in over-smoothing of information during training as our model aggregates information up to the neighbors of its neighbors. Therefore, the most optimal feature for the visibility algorithm is ‘close’ as that is the feature we will be focusing on predicting on the sector-wise model.

For each graph we will consider of window of 30 steps in the data and the graph will be created as illustrated below:
![](https://i.imgur.com/RMqmhit.png)
![](https://i.imgur.com/5PY22Fm.png)
![](https://i.imgur.com/MsleFP8.png)

In this way graphs are created for every window in the dataset for all individual stocks.

### Sector graphs

Sector graphs are inter stock graphs created using the data of stocks belonging to the same sector. For inter stock edges we used the published date included in each node to connect nodes with matching dates as illustrated below:
![](https://i.imgur.com/rbAjchc.png)
![](https://i.imgur.com/9gNLFBO.png)
![](https://i.imgur.com/jbDN3Gx.png)

fig: Graph combining visualization

This process will be done for all the stocks in a sector until all nodes of the same day are connected.

### Edge weights
Edge weights are the degree of connection between the nodes and some GNNs use them for better information pooling from the neighbor nodes. In our graph the weight of the edge between node A and B is $|A_{close}-B_{close}|$. Though most GNNs can learn the weights by themselves prior initialization helps them converge to the optimal weight faster.

## GNN Architectures
In this project we use two architectures **Graph Convolution Network(GCN)** and **Graph Attention Network(GAT)**. These networks though similar have there own pros and cons.

### Graph Convolution Network
![](https://i.imgur.com/6tkjLRQ.png)
This is a three layer architecture with two graph convolutions used to aggregate and update the features of each node using it adjacent neighbors and then its second class neighbors in the first and second layers respectively. In each layer the aggregated features and also fed into a dense layer for forward and back propagation. Then the final layer is used to give desired predictions.

### Graph Attention Network
![](https://i.imgur.com/mZVOfWF.png)
This architecture differs from GCN in that in the aggregation step it introduces attention using the edge weights and only pools information from select neighbors. This is also a three layer architecture with two graph attention layers used to aggregate and update the features of each node using it adjacent neighbors and then its second class neighbors in the first and second layers respectively. In each layer the aggregated features and also fed into a dense layer for forward and back propagation. Then the final layer is used to give desired predictions.

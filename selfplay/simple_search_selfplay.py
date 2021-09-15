import numpy as np
from scipy.spatial.distance import cdist

def find_nearest_node(node, node_seen, method):
    node = np.array(node).reshape(1,-1)
    dist = cdist(node, node_seen, method)

    idx = np.argmin(dist)
    nearest_node = node_seen[idx]

    return list(nearest_node)

## Simple search for M2M(M) data
def simple_search_M(dataset, data_seen, graph, method):
    y_pred = []
    ## Count how many nodes not seen in search graph
    count_oov = 0
    for data in dataset:
        node = str(list(data))
        if graph.has_node(node):
            dialog_acts = [(graph[node][t], t) for t in graph[node]]
            dialog_acts.sort(key=lambda t: t[0]['probability'], reverse=True)
            for dialog_act in dialog_acts[:1]:
                y = eval(dialog_act[1])[19:]
                y_pred.append(y)
        else:
            count_oov += 1

            data_hat = find_nearest_node(data, data_seen, method)
            data_hat = list(map(int, data_hat))
            node = str(data_hat)
            if graph.has_node(node) == False:
                print('incorrect')
            dialog_acts = [(graph[node][t], t) for t in graph[node]]
            dialog_acts.sort(key=lambda t: t[0]['probability'], reverse=True)
            for dialog_act in dialog_acts[:1]:
                y = eval(dialog_act[1])[19:]
                y_pred.append(y)
    return y_pred, count_oov

## Simple search for M2M(R) data
def simple_search_R(dataset, data_seen, graph, method):
    y_pred = []
    ## Count how many nodes not seen in search graph
    count_oov = 0
    for data in dataset:
        node = str(list(data))
        if graph.has_node(node):
            dialog_acts = [(graph[node][t], t) for t in graph[node]]
            dialog_acts.sort(key=lambda t: t[0]['probability'], reverse=True)
            for dialog_act in dialog_acts[:1]:
                y = eval(dialog_act[1])[29:]
                y_pred.append(y)
        else:
            count_oov += 1

            data_hat = find_nearest_node(data, data_seen, method)
            data_hat = list(map(int, data_hat))
            node = str(data_hat)
            if graph.has_node(node) == False:
                print('incorrect')
            dialog_acts = [(graph[node][t], t) for t in graph[node]]
            dialog_acts.sort(key=lambda t: t[0]['probability'], reverse=True)
            for dialog_act in dialog_acts[:1]:
                y = eval(dialog_act[1])[29:]
                y_pred.append(y)
    return y_pred, count_oov

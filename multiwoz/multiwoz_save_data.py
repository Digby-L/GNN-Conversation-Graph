import os
import random
import numpy as np
import torch.utils.data
from multiwoz.conv_graph_gnn import MultiWozConvGraph_GNN
# from conv_graph_gnn import MultiWozConvGraph_GNN

seed = 123456789
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_CUDNN_DETERMINISM'] = str(1)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

depth_list = {1,2,3}

net_depth = 1

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

## GNN subgraphs data
save_data = False
if net_depth == 1:
    train_graph_GNN = MultiWozConvGraph_GNN(dir_name="./", file_names=['train.json'], net_depth=net_depth)
    dev_graph_GNN = MultiWozConvGraph_GNN(dir_name="./", file_names=['val.json'], net_depth=net_depth)
    test_graph_GNN = MultiWozConvGraph_GNN(dir_name="./", file_names=['test.json'], net_depth=net_depth)
    eval_graph_GNN = MultiWozConvGraph_GNN(dir_name="./", file_names=['train.json', 'val.json', 'test.json'],
                                           net_depth=net_depth)
    train_dev_graph = MultiWozConvGraph_GNN(dir_name="./", file_names=['train.json', 'val.json'], net_depth=net_depth)

    ## save data
    x_train_GNN, y_train_GNN, edge_index_train, x_train_simple = train_graph_GNN.generate_subgraph_data()
    x_dev_GNN, y_dev_GNN, edge_index_dev, x_dev_simple = dev_graph_GNN.generate_subgraph_data()
    x_test_GNN, y_test_GNN, edge_index_test, x_test_simple = test_graph_GNN.generate_subgraph_data()

    np.save('./Data1/x_train_GNN.npy', x_train_GNN)
    np.save('./Data1/y_train_GNN.npy', y_train_GNN)
    np.save('./Data1/edge_index_train.npy', edge_index_train)
    np.save('./Data1/x_train_simple.npy', x_train_simple)

    np.save('./Data1/x_dev_GNN.npy', x_dev_GNN)
    np.save('./Data1/y_dev_GNN.npy', y_dev_GNN)
    np.save('./Data1/edge_index_dev.npy', edge_index_dev)
    np.save('./Data1/x_dev_simple.npy', x_dev_simple)

    np.save('./Data1/x_test_GNN.npy', x_test_GNN)
    np.save('./Data1/y_test_GNN.npy', y_test_GNN)
    np.save('./Data1/edge_index_test.npy', edge_index_test)
    np.save('./Data1/x_test_simple.npy', x_test_simple)


elif net_depth == 2:
    train_graph_GNN = MultiWozConvGraph_GNN(dir_name="./", file_names=['train.json'], net_depth=net_depth)
    dev_graph_GNN = MultiWozConvGraph_GNN(dir_name="./", file_names=['val.json'], net_depth=net_depth)
    test_graph_GNN = MultiWozConvGraph_GNN(dir_name="./", file_names=['test.json'], net_depth=net_depth)
    eval_graph_GNN = MultiWozConvGraph_GNN(dir_name="./", file_names=['train.json', 'val.json', 'test.json'],
                                           net_depth=net_depth)
    train_dev_graph = MultiWozConvGraph_GNN(dir_name="./", file_names=['train.json', 'val.json'], net_depth=net_depth)

    ## save data
    x_train_GNN, y_train_GNN, edge_index_train, x_train_simple = train_graph_GNN.generate_subgraph_data()
    x_dev_GNN, y_dev_GNN, edge_index_dev, x_dev_simple = dev_graph_GNN.generate_subgraph_data()
    x_test_GNN, y_test_GNN, edge_index_test, x_test_simple = test_graph_GNN.generate_subgraph_data()

    np.save('./Data2/x_train_GNN.npy', x_train_GNN)
    np.save('./Data2/y_train_GNN.npy', y_train_GNN)
    np.save('./Data2/edge_index_train.npy', edge_index_train)
    np.save('./Data2/x_train_simple.npy', x_train_simple)

    np.save('./Data2/x_dev_GNN.npy', x_dev_GNN)
    np.save('./Data2/y_dev_GNN.npy', y_dev_GNN)
    np.save('./Data2/edge_index_dev.npy', edge_index_dev)
    np.save('./Data2/x_dev_simple.npy', x_dev_simple)

    np.save('./Data2/x_test_GNN.npy', x_test_GNN)
    np.save('./Data2/y_test_GNN.npy', y_test_GNN)
    np.save('./Data2/edge_index_test.npy', edge_index_test)
    np.save('./Data2/x_test_simple.npy', x_test_simple)

elif net_depth == 3:
    train_graph_GNN = MultiWozConvGraph_GNN(dir_name="./", file_names=['train.json'], net_depth=net_depth)
    dev_graph_GNN = MultiWozConvGraph_GNN(dir_name="./", file_names=['val.json'], net_depth=net_depth)
    test_graph_GNN = MultiWozConvGraph_GNN(dir_name="./", file_names=['test.json'], net_depth=net_depth)
    eval_graph_GNN = MultiWozConvGraph_GNN(dir_name="./", file_names=['train.json', 'val.json', 'test.json'],
                                           net_depth=net_depth)
    train_dev_graph = MultiWozConvGraph_GNN(dir_name="./", file_names=['train.json', 'val.json'], net_depth=net_depth)

    ## save data
    x_train_GNN, y_train_GNN, edge_index_train, x_train_simple = train_graph_GNN.generate_subgraph_data()
    x_dev_GNN, y_dev_GNN, edge_index_dev, x_dev_simple = dev_graph_GNN.generate_subgraph_data()
    x_test_GNN, y_test_GNN, edge_index_test, x_test_simple = test_graph_GNN.generate_subgraph_data()

    np.save('./Data3/x_train_GNN.npy', x_train_GNN)
    np.save('./Data3/y_train_GNN.npy', y_train_GNN)
    np.save('./Data3/edge_index_train.npy', edge_index_train)
    np.save('./Data3/x_train_simple.npy', x_train_simple)

    np.save('./Data3/x_dev_GNN.npy', x_dev_GNN)
    np.save('./Data3/y_dev_GNN.npy', y_dev_GNN)
    np.save('./Data3/edge_index_dev.npy', edge_index_dev)
    np.save('./Data3/x_dev_simple.npy', x_dev_simple)

    np.save('./Data3/x_test_GNN.npy', x_test_GNN)
    np.save('./Data3/y_test_GNN.npy', y_test_GNN)
    np.save('./Data3/edge_index_test.npy', edge_index_test)
    np.save('./Data3/x_test_simple.npy', x_test_simple)
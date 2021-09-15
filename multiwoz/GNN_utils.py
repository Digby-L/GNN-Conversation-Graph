import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
from torch.nn import functional as F
from torch.nn import Sequential as Seq, Linear, ReLU
from torch import Tensor
from torch.utils.dlpack import to_dlpack, from_dlpack

from torch import optim
import torch.utils.data
# from torch.utils.data import TensorDataset, DataLoader

from scipy.spatial.distance import cdist

# import sys
# sys.path.append('../')

from torch_geometric.data import InMemoryDataset, Data, Batch
from torch_geometric.data import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv

seed = 123456789
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_CUDNN_DETERMINISM'] = str(1)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_with_soft_loss = False
## Soft loss training is slow, be patient
max_epochs, max_val_f1, patience = 50, 0, 3
batch_size = 32

## Generate torch.geometric datalist
def CreateDatalist(x_data, y_data, edge_data):
    data_list = []

    num_data = len(x_data)

    for i in range(num_data):
        x = torch.tensor(x_data[i])
        y = torch.tensor(y_data[i])
        edge_index = torch.tensor(edge_data[i]).t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index)
        data_list.append(data)
    return data_list

# train_datalist = CreateDatalist(x_train_GNN, y_train_GNN, edge_index_train)
# dev_datalist = CreateDatalist(x_dev_GNN, y_dev_GNN, edge_index_dev)
# test_datalist = CreateDatalist(x_test_GNN, y_test_GNN, edge_index_test)


def load_checkpoint(model, optimizer, filename='checkpoint'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    return model, optimizer


## Graph attention network for embedding (3 layers)
class GAT_emb3(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT_emb3, self).__init__()

        self.conv1 = GATConv(in_channels, 8, heads=1)
        self.conv2 = GATConv(8 * 1, 8, heads=1, concat=False)
        self.conv3 = GATConv(8 * 1, out_channels, heads=1, concat=False)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = F.log_softmax(x, dim=1)

        ## global pooling
        # x = gap(x, data.batch) ## mean pooling
        x = gmp(x, data.batch)  ## max pooling

        return x


## Graph attention network for embedding (2 layers)
class GAT_emb2(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT_emb2, self).__init__()

        self.conv1 = GATConv(in_channels, 8, heads=2)
        self.conv2 = GATConv(8 * 2, out_channels, heads=1, concat=False)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=-1)

        ## global pooling
        # x = gap(x, data.batch) ## mean pooling
        x = gmp(x, data.batch)  ## max pooling

        return x


## Graph attention network for embedding (1 layer)
class GAT_emb1(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT_emb1, self).__init__()

        self.conv1 = GATConv(in_channels, out_channels, heads=2, concat=False)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index

        x = F.elu(self.conv1(x, edge_index))
        x = F.log_softmax(x, dim=-1)

        ## global mean pooling
        # x = gap(x, data.batch)
        x = gmp(x, data.batch)

        return x


## Graph Sage network for embedding (3 layers)
class SageModel3(torch.nn.Module):
    def __init__(self, in_channels, dim_hidden_sage, out_channels):
        super(SageModel3, self).__init__()
        self.conv1 = SAGEConv(in_channels, dim_hidden_sage)
        self.conv2 = SAGEConv(dim_hidden_sage, 128)
        self.conv3 = SAGEConv(128, out_channels)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = F.log_softmax(x, dim=1)

        ## global pooling
        # x = gap(x, data.batch) ## mean pooling
        x = gmp(x, data.batch)  ## max pooling

        return x


## Graph Sage network for embedding (2 layers)
class SageModel2(torch.nn.Module):
    def __init__(self, in_channels, dim_hidden_sage, out_channels):
        super(SageModel2, self).__init__()
        self.conv1 = SAGEConv(in_channels, dim_hidden_sage)
        self.conv2 = SAGEConv(dim_hidden_sage, out_channels)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)

        ## global pooling
        # x = gap(x, data.batch) ## mean pooling
        x = gmp(x, data.batch)  ## max pooling

        return x


## Graph Sage network for embedding (1 layer)
class SageModel1(torch.nn.Module):
    def __init__(self, in_channels, dim_hidden_sage, out_channels):
        super(SageModel1, self).__init__()
        self.conv1 = SAGEConv(in_channels, out_channels)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index

        x = self.conv1(x, edge_index)
        x = F.log_softmax(x, dim=1)

        ## global mean pooling
        # x = gap(x, data.batch)
        x = gmp(x, data.batch)

        return x


## Single-layer MLP with global sum pooling
class MLPClassifier(torch.nn.Module):

    def __init__(self, emb_model, dim_input, dim_target, dim_hidden):
        super(MLPClassifier, self).__init__()
        '''
        dim_input: dimension of node feature (355)
        dim_target: dimension of target y (309)
        dim_hidden: number of hidden nodes
        max_num_nodes: max number of nodes in subgraphs
        '''

        self.emb = emb_model
        self.fc_global = Linear(dim_input, dim_hidden)
        self.out = Linear(dim_hidden, dim_target)

    def forward(self, data):
        x = self.emb(data)
        x = self.out(F.relu(self.fc_global(x)))

        return x


## Define validation and evaluation functions
def validate_model_GNN(model, dataloader, convgraph, dim_target, train_with_soft_loss):
    model.eval()
    if train_with_soft_loss:
        loss_function = SoftBCEWithLogitsLoss()
    else:
        loss_function = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        valid_loss = 0
        valid_f1s = []
        valid_losses = []

        for batch in dataloader:
            output = model(batch)
            batch_size = batch.num_graphs

            labels = batch.y.reshape(batch_size, dim_target)

            for out, lab in zip(output, labels):
                valid_f1s.append(f1((out > 0.0).int(), lab).cpu())

            if train_with_soft_loss:
                loss = loss_function(output.float(), batch.x.float(), convgraph)
            else:
                loss = loss_function(output.float(), labels.float())

            valid_losses.append(loss.data.mean().cpu())

    model.train()
    return np.mean(valid_losses), np.mean(valid_f1s)



def validate_model_GNN_soft(model, dataloader, convgraph, evel_graph, data_simple, dim_target, train_with_soft_loss, valid_with_soft_F1):
    model.eval()
    if train_with_soft_loss:
        loss_function = SoftBCEWithLogitsLoss()
    else:
        loss_function = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        valid_loss = 0
        valid_f1s = []
        valid_losses = []

        count = 1
        for batch in dataloader:
            output = model(batch)
            batch_size = batch.num_graphs

            inputs = data_simple[(count - 1) * batch_size:count * batch_size]
            labels = batch.y.reshape(batch_size, dim_target)

            count += 1

            if valid_with_soft_F1:
                for inp, out, lab in zip(inputs, output, labels):
                    y_pred = (out > 0.0).int()
                    nearest_gold, best_f1 = evel_graph.get_best_f1_score(inp.data.tolist(), y_pred.data.tolist())
                    valid_f1s.append(best_f1)
            else:
                for out, lab in zip(output, labels):
                    valid_f1s.append(f1((out > 0.0).int(), lab).cpu())

            if train_with_soft_loss:
                loss = loss_function(output.float(), batch.x.float(), convgraph)
            else:
                loss = loss_function(output.float(), labels.float())

            valid_losses.append(loss.data.mean().cpu())

    model.train()
    return np.mean(valid_losses), np.mean(valid_f1s)


def evaluate_model_GNN(model, dataloader, data_simple, eval_graph, dim_target, report=False):
    model.eval()
    with torch.no_grad():
        gold_labels = []
        nearest_gold_labels = []
        pred_labels = []
        soft_f1s = []

        count = 1
        for batch in dataloader:
            output = model(batch)
            batch_size = batch.num_graphs

            inputs = data_simple[(count - 1) * batch_size:count * batch_size]
            labels = batch.y.reshape(batch_size, dim_target)

            count += 1

            for inp, out, lab in zip(inputs, output, labels):
                y_pred = (out > 0.0).int()
                pred_labels.append(y_pred.cpu().numpy())
                gold_labels.append(lab.cpu().numpy())

                nearest_gold, best_f1 = eval_graph.get_best_f1_score(inp.data.tolist(), y_pred.data.tolist())
                # nearest_gold, best_f1 = get_best_f1_score(inp.data.tolist(), y_pred.data.tolist())

                soft_f1s.append(best_f1)
                nearest_gold_labels.append(nearest_gold)

        print("Hard F-Score (exact match): %.3f" % f1_score(y_true=np.array(gold_labels, dtype='float32'),
                                                            y_pred=np.array(pred_labels, dtype='float32'),
                                                            average='samples'))
        print("Soft F-Score (best match): %f" % np.mean(soft_f1s))
        if report:
            print(classification_report(y_true=np.array(nearest_gold_labels, dtype='float32'),
                                        y_pred=np.array(pred_labels, dtype='float32'),
                                        target_names=eval_graph.dialog_act_to_idx.keys(), digits=3))
    model.train()


#### Sequential Models

class LSTM_model(nn.Module):
    ## [CODES NOT AVAILABLE FOR THIS PART]


class Perceptron(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Perceptron, self).__init__()
        # self.fc1 = nn.Linear(4*input_size, output_size)
        # self.fc2
        # self.relu = torch.nn.ReLU()
        self.fc_global = nn.Linear(input_size * 4, 128)
        self.out = nn.Linear(128, output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        # output = self.fc(x)
        # output = self.relu(output) # instead of Heaviside step fn
        x = self.out(F.relu(self.fc_global(x)))
        return x


class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.hidden_to_logits = nn.Linear(hidden_size*2, output_size)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        out, _ = self.lstm(inputs)
        logits = self.hidden_to_logits(self.relu(out[:, -1, :]))
        return logits

class RNN_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_model, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.hidden_to_logits = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        out, _ = self.rnn(inputs)
        logits = self.hidden_to_logits(self.relu(out[:, -1, :]))
        return logits


class SoftBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    __constants__ = ['weight', 'pos_weight', 'reduction']
    ## [CODES NOT AVAILABLE FOR THIS PART]

def f1(y_pred, y_true):
    ## [CODES NOT AVAILABLE FOR THIS PART]


def validate_model(classifier, data_generator, convgraph, train_with_soft_loss, device):
    ## [CODES NOT AVAILABLE FOR THIS PART]


def evaluate_model(classifier, data_generator, eval_graph, device, report=False):
    ## [CODES NOT AVAILABLE FOR THIS PART]


##### Simple Search Method
def find_nearest_node(node, node_seen, method):
    node = np.array(node).reshape(1,-1)
    dist = cdist(node, node_seen, method)

    idx = np.argmin(dist)
    nearest_node = node_seen[idx]

    return list(nearest_node)


def simple_search(dataset, data_seen, graph, method):
    y_pred = []
    ## Count how many nodes not seen in search graph
    count_oov = 0
    for data in dataset:
        node = str(list(data))
        if graph.has_node(node):
            dialog_acts = [(graph[node][t], t) for t in graph[node]]
            dialog_acts.sort(key=lambda t: t[0]['probability'], reverse=True)
            for dialog_act in dialog_acts[:1]:
                y = eval(dialog_act[1])[46:]
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
                y = eval(dialog_act[1])[46:]
                y_pred.append(y)
    return y_pred, count_oov



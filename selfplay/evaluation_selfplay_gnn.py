# coding=utf-8
import os
import random
import numpy as np
from torch import optim
import torch.nn as nn
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import DataLoader as GDataloader

from conv_graph_selfplay_gnn import SelfPlayConvGraph
from multiwoz.GNN_utils import evaluate_model, LSTM_model, Perceptron, BiLSTM, RNN_model
from multiwoz.GNN_utils import SoftBCEWithLogitsLoss, validate_model, f1, load_checkpoint
from multiwoz.GNN_utils import CreateDatalist, GAT_emb3, GAT_emb2, GAT_emb1, SageModel3, SageModel2, SageModel1, MLPClassifier
from multiwoz.GNN_utils import validate_model_GNN, evaluate_model_GNN, validate_model_GNN_soft
from selfplay.simple_search_selfplay import find_nearest_node, simple_search_M, simple_search_R

seed = 123456789
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_CUDNN_DETERMINISM'] = str(1)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


history = 4
## default history is 4
dataset = "restaurant"
## dataset options = 'movie' or 'restaurant'
train_with_soft_loss = False
## soft loss training is slow, be patient
max_epochs, max_val_f1, patience = 50, 0, 5
net_depth = 1

## --------------------------------------------------------------------
train_graph_GNN = SelfPlayConvGraph(dir_name="./selfplay/" + dataset, file_names=['/train.json'], seq_length=history, net_depth = net_depth)
dev_graph_GNN = SelfPlayConvGraph(dir_name="./selfplay/" + dataset, file_names=["/dev.json"], seq_length=history, net_depth = net_depth)
test_graph_GNN = SelfPlayConvGraph(dir_name="./selfplay/" + dataset, file_names=["/test.json"], seq_length=history, net_depth = net_depth)
eval_graph_GNN = SelfPlayConvGraph(dir_name="./selfplay/" + dataset, file_names=['/train.json', "/dev.json", "/test.json"], seq_length=history, net_depth = net_depth)

## Construct subgraphs
x_train_GNN, y_train_GNN, edge_index_train, x_train_simple = train_graph_GNN.generate_subgraph_data()
x_dev_GNN, y_dev_GNN, edge_index_dev, x_dev_simple = dev_graph_GNN.generate_subgraph_data()
x_test_GNN, y_test_GNN, edge_index_test, x_test_simple = test_graph_GNN.generate_subgraph_data()

# max_num_nodes_train = max([len(x_train_GNN[i]) for i in range(len(x_train_GNN))])
# max_num_nodes_test = max([len(x_test_GNN[i]) for i in range(len(x_test_GNN))])
# max_num_nodes_dev = max([len(x_dev_GNN[i]) for i in range(len(x_dev_GNN))])
#
# print(max_num_nodes_train, max_num_nodes_test, max_num_nodes_dev)
#
# min_num_nodes_train = min([len(x_train_GNN[i]) for i in range(len(x_train_GNN))])
# min_num_nodes_test = min([len(x_test_GNN[i]) for i in range(len(x_test_GNN))])
# min_num_nodes_dev = min([len(x_dev_GNN[i]) for i in range(len(x_dev_GNN))])
#
# print(min_num_nodes_train, min_num_nodes_test, min_num_nodes_dev)

## Generate test data for simple search in search_graph
train_graph = train_graph_GNN.graph
# search_graph = train_dev_graph.graph
x_test_search = [x_test_GNN[i][-1] for i in range(len(x_test_GNN))]
# x_test_unique = np.unique(x_test, axis = 0)

x_dev_search = [x_dev_GNN[i][-1] for i in range(len(x_dev_GNN))]
# x_dev_unique = np.unique(x_dev, axis = 0)

x_seen_train = [x_train_simple[i][-1] for i in range(len(x_train_simple))]
x_seen_dev = [x_dev_GNN[i][-1] for i in range(len(x_dev_GNN))]
x_seen = np.concatenate((x_seen_train, x_seen_dev))

## Create graph datalist
train_datalist = CreateDatalist(x_train_GNN, y_train_GNN, edge_index_train)
dev_datalist = CreateDatalist(x_dev_GNN, y_dev_GNN, edge_index_dev)
test_datalist = CreateDatalist(x_test_GNN, y_test_GNN, edge_index_test)

## Batch data droplast
batch_size = 32
train_loader = GDataloader(train_datalist, batch_size=batch_size, drop_last=True)
dev_loader = GDataloader(dev_datalist, batch_size=batch_size, drop_last=True)
test_loader = GDataloader(test_datalist, batch_size=batch_size, drop_last=True)

print("Batch created successfully.")

## Train loss
train_with_soft_loss = False

if train_with_soft_loss:
    loss_function = SoftBCEWithLogitsLoss()
else:
    loss_function = nn.BCEWithLogitsLoss()


## Model parameters
## Graph attention
## For MultiWOZ

# dim_input = len(train_graph_GNN.belief_state_to_idx) + len(train_graph_GNN.dialog_act_to_idx)
# dim_target = len(train_graph_GNN.dialog_act_to_idx)

dim_input = torch.tensor(x_train_GNN[1]).shape[1]
dim_target = torch.tensor(y_train_GNN[1]).shape[0]
dim_hidden_sage = 256
dim_hidden = 256
params = {'batch_size': 32, 'shuffle': True}

## Choose model
model_name = ['GAT', 'GSage', 'MeanPool', 'LinearData']
emb_model_name = 'GAT'

## Graph attention or graph sage
if emb_model_name == 'GAT':
    if net_depth == 3:
        emb_model = GAT_emb3(dim_input, dim_input)
        model = MLPClassifier(emb_model, dim_input, dim_target, dim_hidden=256)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    elif net_depth == 2:
        emb_model = GAT_emb2(dim_input, dim_input)
        model = MLPClassifier(emb_model, dim_input, dim_target, dim_hidden=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif net_depth == 1:
        emb_model = GAT_emb1(dim_input, dim_input)
        model = MLPClassifier(emb_model, dim_input, dim_target, dim_hidden=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    max_epochs, max_val_f1, patience = 300, 0, 30

elif emb_model_name == 'GSage':
    if net_depth == 3:
        emb_model = SageModel3(dim_input, dim_hidden_sage, dim_input)
        model = MLPClassifier(emb_model, dim_input, dim_target, dim_hidden=128)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif net_depth == 2:
        emb_model = SageModel2(dim_input, dim_hidden_sage, dim_input)
        model = MLPClassifier(emb_model, dim_input, dim_target, dim_hidden=256)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif net_depth == 1:
        emb_model = SageModel1(dim_input, dim_hidden_sage, dim_input)
        model = MLPClassifier(emb_model, dim_input, dim_target, dim_hidden=256)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    max_epochs, max_val_f1, patience = 300, 0, 30

elif emb_model_name == 'MeanPool':
    classifier = LSTM_model(dim_input, dim_target)
    optimizer = optim.RMSprop(classifier.parameters())

    training_set = TensorDataset(torch.tensor(x_train_simple, dtype=torch.float32), torch.tensor(y_train_GNN, dtype=torch.float32))
    training_generator = DataLoader(training_set, **params)
    validation_set = TensorDataset(torch.tensor(x_dev_simple, dtype=torch.float32), torch.tensor(y_dev_GNN, dtype=torch.float32))
    validation_generator = DataLoader(validation_set, **params)
    no_dupl_test_set = TensorDataset(torch.tensor(x_test_simple, dtype=torch.float32),
                                     torch.tensor(y_test_GNN, dtype=torch.float32))
    no_dupl_test_generator = DataLoader(no_dupl_test_set, **params)

    max_epochs, max_val_f1, patience = 50, 0, 5

elif emb_model_name == 'LinearData':
    x_train, y_train = train_graph_GNN.generate_standard_data(unique=False)
    x_dev, y_dev = dev_graph_GNN.generate_standard_data(unique=False)
    x_test, y_test = test_graph_GNN.generate_standard_data(unique=True)

    # classifier = LSTM_model(dim_input, dim_target)
    classifier = Perceptron(dim_input, dim_target)
    # classifier = BiLSTM(dim_input, dim_hidden, dim_target)
    # classifier = RNN_model(dim_input, dim_hidden, dim_target)

    optimizer = optim.RMSprop(classifier.parameters())
    training_set = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    training_generator = DataLoader(training_set, **params)
    validation_set = TensorDataset(torch.tensor(x_dev, dtype=torch.float32), torch.tensor(y_dev, dtype=torch.float32))
    validation_generator = DataLoader(validation_set, **params)
    no_dupl_test_set = TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                                     torch.tensor(y_test, dtype=torch.float32))
    no_dupl_test_generator = DataLoader(no_dupl_test_set, **params)

    max_epochs, max_val_f1, patience = 50, 0, 5


## Model training for GraphData
# for epoch in range(max_epochs):
#     f1s = []
#     losses = []
#     for batch in train_loader:
#         optimizer.zero_grad()
#
#         outputs = model(batch)
#         batch_size = batch.num_graphs
#
#         labels = batch.y.reshape(batch_size, dim_target)
#         inputs = batch.x
#
#         if train_with_soft_loss:
#             loss = loss_function(outputs.float(), inputs.float(), train_graph_GNN)
#         else:
#             loss = loss_function(outputs.float(), labels.float())
#
#         loss.backward()
#         optimizer.step()
#         losses.append(loss.data.mean().cpu())
#
#         for out, lab in zip(outputs, labels):
#             f1s.append(f1((out > 0.0).float(), lab).cpu())
#
#     valid_loss, valid_f1 = validate_model_GNN(model, dev_loader, dev_graph_GNN, dim_target, train_with_soft_loss)
#     # valid_loss, valid_f1 = validate_model_GNN_soft(model, dev_loader, None, eval_graph_GNN, x_dev_simple, dim_target,
#     #                                           train_with_soft_loss, True)
#
#     # noinspection PyStringFormat
#     print('[%d/%d] Train Loss: %.3f, Train F1: %.3f, Val Loss: %.3f, Val F1: %.3f,' % (
#     epoch + 1, 300, np.mean(losses), np.mean(f1s), valid_loss, valid_f1))
#
#     # Early stopping
#     if valid_f1 > max_val_f1:
#         state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), }
#         torch.save(state, 'checkpoint')
#
#         max_val_f1 = valid_f1
#         max_epochs = 0
#     else:
#         max_epochs += 1
#         if max_epochs >= patience:
#             model, optimizer = load_checkpoint(model, optimizer)
#             print("Stopped early and went back to Validation f1: %.3f" % max_val_f1)
#             break
#
#
# print("---------------------- DEVELOPMENT SET REPORT --------------------------")
# evaluate_model_GNN(model, dev_loader, x_dev_simple, eval_graph_GNN, dim_target)
#
# print("--------------------- DEDUPLICATED TEST SET REPORT -------------------------")
# evaluate_model_GNN(model, test_loader, x_test_simple, eval_graph_GNN, dim_target)
#


##############################################
##############################################

## Training for MeanData and LinearData

# for epoch in range(max_epochs):
#     f1s = []
#     losses = []
#     for inputs, labels in training_generator:
#         classifier.zero_grad()
#
#         inputs, labels = inputs.to(device), labels.to(device)
#         output = classifier(inputs)
#         if train_with_soft_loss:
#             loss = loss_function(output, inputs, train_graph_GNN)
#         else:
#             loss = loss_function(output, labels)
#
#         loss.backward()
#         optimizer.step()
#         losses.append(loss.data.mean().cpu())
#
#         for out, lab in zip(output, labels):
#             f1s.append(f1((out > 0.0).float(), lab).cpu())
#
#     valid_loss, valid_f1 = validate_model(classifier, validation_generator, dev_graph_GNN, train_with_soft_loss, device)
#     # noinspection PyStringFormat
#     print('[%d/%d] Train Loss: %.3f, Train F1: %.3f, Val Loss: %.3f, Val F1: %.3f,' % (epoch + 1, 50, np.mean(losses), np.mean(f1s), valid_loss, valid_f1))
#
#     # Early stopping
#     if valid_f1 > max_val_f1:
#         state = {'epoch': epoch + 1, 'state_dict': classifier.state_dict(), 'optimizer': optimizer.state_dict(), }
#         torch.save(state, 'checkpoint')
#
#         max_val_f1 = valid_f1
#         max_epochs = 0
#     else:
#         max_epochs += 1
#         if max_epochs >= patience:
#             classifier, optimizer = load_checkpoint(classifier, optimizer)
#             print("Stopped early and went back to Validation f1: %.3f" % max_val_f1)
#             break
#
# print("---------------------- DEVELOPMENT SET REPORT --------------------------")
# evaluate_model(classifier, validation_generator, eval_graph_GNN, device)
#
# print("--------------------- DEDUPLICATED TEST SET REPORT -------------------------")
# evaluate_model(classifier, no_dupl_test_generator, eval_graph_GNN, device)



#####
##### Simple Search Method
use_search = True
## method_list = {'euclidean', 'jaccard'}
method = 'jaccard'

if use_search == True:
    f1s_test = []
    ## Test set results
    if dataset == "restaurant":
        outputs, test_oov = simple_search_R(x_test_search, x_seen_train, train_graph, method)
    elif dataset == "movie":
        outputs, test_oov = simple_search_M(x_test_search, x_seen_train, train_graph, method)

    outputs = torch.tensor(outputs).float()
    labels = torch.tensor(y_test_GNN).float()

    inputs = x_test_simple

    soft_f1s = []

    if train_with_soft_loss:
        loss_test = loss_function(outputs, x_test_search, train_graph_GNN)
    else:
        loss_test = loss_function(outputs, labels)

    for out, lab in zip(outputs, labels):
        f1s_test.append(f1((out > 0.0).float(), lab).cpu())

    for inp, out, lab in zip(inputs, outputs, labels):
        y_pred = (out > 0.0).int()
        nearest_gold, best_f1 = eval_graph_GNN.get_best_f1_score(inp.data.tolist(), y_pred.data.tolist())

        soft_f1s.append(best_f1)

    print('Testset result of Simple Search Method:', 'Test loss:', loss_test.data.mean().cpu(), 'Test F1:', np.mean(f1s_test))
    print('Number of test nodes not found in the search graph:', test_oov)
    print('Testset Soft F1: ', np.mean(soft_f1s))

    ##################################
    ## Dev set results
    f1s_dev = []
    soft_f1s_dev = []
    if dataset == "restaurant":
        outputs_dev, dev_oov = simple_search_R(x_dev_search, x_seen_train, train_graph, method)
    elif dataset == "movie":
        outputs_dev, dev_oov = simple_search_M(x_dev_search, x_seen_train, train_graph, method)

    outputs_dev = torch.tensor(outputs_dev).float()
    labels_dev = torch.tensor(y_dev_GNN).float()

    inputs_dev = x_dev_simple

    if train_with_soft_loss:
        loss_dev = loss_function(outputs_dev, x_dev_search, train_graph_GNN)
    else:
        loss_dev = loss_function(outputs_dev, labels_dev)

    for out, lab in zip(outputs_dev, labels_dev):
        f1s_dev.append(f1((out > 0.0).float(), lab).cpu())

    for inp, out, lab in zip(inputs_dev, outputs_dev, labels_dev):
        y_pred = (out > 0.0).int()
        nearest_gold, best_f1 = eval_graph_GNN.get_best_f1_score(inp.data.tolist(), y_pred.data.tolist())

        soft_f1s_dev.append(best_f1)

    print('Devset result of Simple Search Method:', 'Valid loss:', loss_dev.data.mean().cpu(), 'Valid F1:', np.mean(f1s_dev))
    print('Number of dev nodes not found in the search graph:', dev_oov)
    print('Testset Soft F1: ', np.mean(soft_f1s_dev))
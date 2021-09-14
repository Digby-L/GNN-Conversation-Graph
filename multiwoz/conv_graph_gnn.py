# coding=utf-8
import json
from os import listdir
from os.path import isfile, join
import networkx as nx
from sklearn.metrics import f1_score
from typing import Dict, List, Tuple
import numpy
import matplotlib.pyplot as plt
import torch
import torch_geometric.data
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import DataLoader



class MultiWozConvGraph_GNN:

    def __init__(self, dir_name: str, file_names: List[str], seq_length: int = 4, net_depth: int = 3):
        self.seq_length = seq_length
        self.dialogue_id = 0
        self.dir_name = dir_name
        self.file_names = file_names
        self.graph = nx.DiGraph()
        self._initiate_graph()
        self.net_depth = net_depth
        self.visited_sub_nodes = set()

    def _initiate_graph(self) -> None:
        ## [CODES NOT AVAILABLE FOR THIS PART]

    def _init_graph_vectors(self) -> List[Dict[str, int]]:
        ## [CODES NOT AVAILABLE FOR THIS PART]

    def generate_subgraph_data(self, padding = False, simple_mean = True, to_json: bool = False):
        self.augmented_paths.clear()
        """
        simple_mean: True [use mean pooling for each layer]
        self.net_depth: depth of each subgraph
        visited_sub_nodes: nodes whose subgraph is already created
        subgraph_list: nodes in the subgraph of each visited_sub_nodes
        SUB_G: a new graph containing all subgraphs disjointly
        
        return:
        train_x: nodes in each subgraph
        train_y: y of each subgraph
        train_edge_index: edge_index of each subgraph
        """
        G = self.graph.copy(as_view=False)
        G_reverse = G.reverse()
        self.visited_sub_nodes.clear()

        # subgraph_list = []
        # SUB_G = nx.DiGraph()

        train_x = []
        train_y = []
        train_edge_index = []
        num_subgraph = 0

        train_x_simple = []

        for f_name in self.file_names:
            with open(self.dir_name + f_name, 'r') as f:
                data = json.load(f)
                ## [CODES HIDDEN FOR THIS PART]

                    if not to_json:
                        for i in range(2, len(visited_nodes) - 1, 2):
                            current_node = visited_nodes[i-1]
                            # current_node = visited_nodes[i]
                            # x = visited_nodes[max(0, i - self.seq_length): i]
                            y = visited_nodes[i][len(self.belief_state_to_idx):]

                            ## Find list of ancestors of the current node with given depth
                            neighbour_list = list(nx.dfs_predecessors(G_reverse, str(current_node), depth_limit=self.net_depth))

                            ## Add the current_node to neighbour_list and change string to list
                            nei_extend_list = neighbour_list + [str(current_node)]
                            x = list(eval(i) for i in nei_extend_list)

                            x = numpy.array(x, dtype='b')

                            train_x.append(x)
                            train_y.append(y)

                            ## Extract subgraph from G to get subgraph's edge index
                            dialogue_subgraph = G.subgraph(nei_extend_list)

                            ## Count number of subgraphs created
                            num_subgraph += 1

                            ## Create simple data by averaging each hop
                            simple_data = []
                            simple_data.append(current_node)
                            # node = str(nei_extend_list[-1])
                            node = str(current_node)
                            for depth in range(1, self.net_depth+1):
                                nei_dep = self.kneighbours(dialogue_subgraph, [node], depth)
                                if len(nei_dep) != 0:
                                    x_simple = list(eval(i) for i in nei_dep)
                                    simple_data.insert(0, numpy.mean(x_simple, axis = 0))
                                else:
                                    simple_data.insert(0, [0] * (len(self.belief_state_to_idx) + len(self.dialog_act_to_idx)))

                            simple_data = numpy.array(simple_data, dtype='b')
                            train_x_simple.append(simple_data)


                            ## Change edge index to matrix
                            # sub_edge = [list(x) for x in list(dialogue_subgraph.edges())]
                            # sub_edge_index = list(list(eval(sub_edge[j][i]) for i in range(len(sub_edge[j]))) for j in range(len(sub_edge)))
                            # train_edge_index.append(sub_edge_index)
                            # self.visited_sub_nodes.add(str(current_node))
                            ### Draw the subgraph
                            # pos = nx.spring_layout(dialogue_subgraph)
                            # nx.draw(dialogue_subgraph, pos)
                            # plt.show()

                            ### Relabel the node index to integers
                            # SUB_G = nx.DiGraph()
                            # SUB_G = nx.disjoint_union(SUB_G, dialogue_subgraph)

                            SUB_G = nx.relabel.convert_node_labels_to_integers(dialogue_subgraph)
                            sub_edge = [list(x) for x in list(SUB_G.edges())]
                            train_edge_index.append(sub_edge)


        # if padding:
        #     ## Zero-padding for train_x
        #     max_len_x = max([len(train_x[i]) for i in range(len(train_x))])
        #     seq_len = len(train_x[0][1])
        #     for i in range(len(train_x)):
        #         while len(train_x[i]) < max_len_x:
        #             train_x[i].append([0] * seq_len)
        #
        #     ## Zero-padding for edge_index
        #     max_len_edge = max([len(train_edge_index[i]) for i in range(len(train_edge_index))])
        #     for i in range(len(train_edge_index)):
        #         while len(train_edge_index[i]) < max_len_edge:
        #             train_edge_index[i].append([[0]*seq_len, [0]*seq_len])


        print("-----------------------------------------------")
        print("Stats for ConvGraph for %s%s" % (self.dir_name, " and ".join(self.file_names)))
        print("Subgraph list created successfully.")
        print("Number of subgraphs created: %d" % num_subgraph)
        print("-----------------------------------------------")

        return train_x, train_y, train_edge_index, train_x_simple

    def kneighbours(self, g, start_node, depth):
        neighbours = start_node
        for l in range(depth):
            # neighbours = list(neigh for n in neighbours for neigh in g[n])
            neighbours = list(neigh for n in neighbours for neigh in g.predecessors(n))
        return neighbours

    def generate_augmented_data(self, to_json: bool = False) -> numpy.array:
        ## [CODES NOT AVAILABLE FOR THIS PART]

    def generate_standard_data(self, unique: bool) -> numpy.array:
    ## [CODES NOT AVAILABLE FOR THIS PART]

    def get_valid_dialog_acts(self, state: List[List[float]]):
        ## [CODES NOT AVAILABLE FOR THIS PART]

    def get_best_f1_score(self, state: List[List[float]], y_pred: List[float]) -> Tuple[List[int], float]:
        ## [CODES NOT AVAILABLE FOR THIS PART]


# train_graph = MultiWozConvGraph_GNN(dir_name="./", file_names=['train.json'])
# train_graph.generate_standard_data(unique=False)
# train_graph.generate_augmented_data(to_json=True)

# train_graph_GNN = MultiWozConvGraph_GNN(dir_name="./", file_names=['train.json'])
# x_train_GNN, y_train_GNN, edge_index_train, x_train_simple = train_graph_GNN.generate_subgraph_data()
# print('sub_x', x_train_simple[1][-1])
# print('sub_y', y_train_GNN[1])

# x, y = train_graph_GNN.generate_standard_data(unique=False)
# print('stan_x', x[1][-1])
# print('stan_y', y[1])

# print('Train subgraph created.')

# train_graph_GNN = MultiWozConvGraph_GNN(dir_name="./", file_names=['train.json'], net_depth=3)
# dev_graph_GNN = MultiWozConvGraph_GNN(dir_name="./", file_names=['val.json'], net_depth=3)
# test_graph_GNN = MultiWozConvGraph_GNN(dir_name="./", file_names=['test.json'], net_depth=3)
# eval_graph_GNN = MultiWozConvGraph_GNN(dir_name="./", file_names=['train.json', 'val.json', 'test.json'], net_depth=3)

# ## GNN subgraphs data
# x_train_GNN, y_train_GNN, edge_index_train, x_train_simple = train_graph_GNN.generate_subgraph_data()
# x_dev_GNN, y_dev_GNN, edge_index_dev, x_dev_simple = dev_graph_GNN.generate_subgraph_data()
# x_test_GNN, y_test_GNN, edge_index_test, x_test_simple = test_graph_GNN.generate_subgraph_data()

### Save data test
# numpy.save('./temp_data/x_train_GNN.npy', x_train_GNN)
# numpy.save('./temp_data/y_train_GNN.npy', y_train_GNN)
# numpy.save('./temp_data/edge_index_train.npy', edge_index_train)
# numpy.save('./temp_data/x_train_simple.npy', x_train_simple)

# numpy.save('./Data/x_dev_GNN.npy', x_dev_GNN)
# numpy.save('./Data/y_dev_GNN.npy', y_dev_GNN)
# numpy.save('./Data/edge_index_dev.npy', edge_index_dev)
# numpy.save('./Data/x_dev_simple.npy', x_dev_simple)

# numpy.save('./Data/x_test_GNN.npy', x_test_GNN)
# numpy.save('./Data/y_test_GNN.npy', y_test_GNN)
# numpy.save('./Data/edge_index_test.npy', edge_index_test)
# numpy.save('./Data/x_test_simple.npy', x_test_simple)


# print([len(train_x[i]) for i in range(len(train_x))])
# print(len(train_x))
# print(len(train_edge_index))
# print([len(train_edge_index[1][1][i]) for i in range(len(train_edge_index[1][1]))])

# temp = Data(x = train_x[1], y = train_y[1], edge_index = torch.tensor(train_edge_index[1]))

###############################
##### TEST RUN FOR nx.dfs #####
###############################
H = nx.DiGraph()
H.add_edges_from([('E', 'A'), ('A', 'M'), ('A', 'B'), ('B', 'C')])
H.add_edges_from([('F', 'D'), ('D', 'B')])
H.add_edges_from([('P', 'M'), ('M', 'N'), ('N', 'R'), ('Q', 'N')])
H.add_edges_from([('D', 'C'), ('A', 'R')])
# nx.draw(H, with_labels = True)

a = list(nx.dfs_predecessors(H.reverse(), 'C', depth_limit = 0))
# print(a)
## >> ['B']

b = list(nx.dfs_predecessors(H.reverse(), 'C', depth_limit = 1))
# print(b)
## >> ['B']

c = list(nx.dfs_predecessors(H.reverse(), 'C', depth_limit = 2))
# print(c)
## >> ['B', 'A', 'D']

d = list(nx.dfs_predecessors(H.reverse(), 'C', depth_limit = 3))
# print(d)
## >> ['B', 'A', 'E', 'D', 'F']

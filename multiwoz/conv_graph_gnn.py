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
        self.augmented_conversations = dict()
        self.augmented_paths = set()
        g_vectors = self._init_graph_vectors()
        self.belief_state_to_idx = g_vectors[0]
        self.dialog_act_to_idx = g_vectors[1]
        self.final_state = str([1] * (len(self.belief_state_to_idx) + len(self.dialog_act_to_idx)))
        self._initiate_graph()
        self.net_depth = net_depth
        self.visited_sub_nodes = set()

    def _initiate_graph(self) -> None:
        no_of_dialogues = 0
        unique_turns, repetitive_turns = 0.0, 0.0
        for f_name in self.file_names:
            with open(self.dir_name + f_name, 'r') as f:
                data = json.load(f)
                for dialogue in data:
                    dialogue = data[dialogue]
                    dialogue['goal'] = {}
                    no_of_dialogues += 1
                    last_belief_state = [0] * len(self.belief_state_to_idx)
                    previous_state = [0] * (len(self.belief_state_to_idx) + len(self.dialog_act_to_idx))
                    for turn in dialogue['log']:
                        if len(turn['metadata']) > 0:
                            current_state = [0] * (len(self.belief_state_to_idx) + len(self.dialog_act_to_idx))
                            for domain in turn['metadata']:
                                for slot in turn['metadata'][domain]['semi']:
                                    if turn['metadata'][domain]['semi'][slot] not in ["", "not mentioned", "none"]:
                                        index = self.belief_state_to_idx[domain + "_" + slot.lower()]
                                        current_state[index] = 1
                                        last_belief_state[index] = 1
                                for slot in turn['metadata'][domain]['book']:
                                    if slot == "booked":
                                        for item in turn['metadata'][domain]['book'][slot]:
                                            for key in item:
                                                index = self.belief_state_to_idx[
                                                    domain + "_" + slot + "_" + key.lower()]
                                                current_state[index] = 1
                                                last_belief_state[index] = 1
                                    else:
                                        if turn['metadata'][domain]['book'][slot] not in ["", "not mentioned", "none"]:
                                            index = self.belief_state_to_idx[domain + "_" + slot.lower()]
                                            current_state[index] = 1
                                            last_belief_state[index] = 1
                            if len(turn['dialog_act']) == 0:
                                index = self.dialog_act_to_idx["empty_dialogue_act"]
                                current_state[index + len(self.belief_state_to_idx)] = 1
                            else:
                                for act in turn['dialog_act']:
                                    for slot in turn['dialog_act'][act]:
                                        index = self.dialog_act_to_idx[act.lower() + "_" + slot[0].lower()]
                                        current_state[index + len(self.belief_state_to_idx)] = 1
                                    index = self.dialog_act_to_idx[act.lower()]
                                    current_state[index + len(self.belief_state_to_idx)] = 1
                            start, end = str(previous_state), str(current_state)
                            ## Check if the graph already contains this turn
                            ## Turns aiming the same slot and value will be the same in the graph
                            ## graph.has_edge returns True if the edge exits in the graph
                            if self.graph.has_edge(start, end):
                                self.graph[start][end]['turn'].append(turn)
                                self.graph[start][end]['probability'] += 1.0
                                repetitive_turns += 1
                            else:
                                self.graph.add_edge(start, end, turn=[turn], probability=1.0)
                                unique_turns += 1
                            previous_state = current_state
                        ## else here: len(turn['metadata']) = 0
                        else:
                            current_state = last_belief_state + ([0] * len(self.dialog_act_to_idx))
                            if len(turn['dialog_act']) == 0:
                                index = self.dialog_act_to_idx["empty_dialogue_act"]
                                current_state[index + len(self.belief_state_to_idx)] = 1
                            else:
                                for act in turn['dialog_act']:
                                    for slot in turn['dialog_act'][act]:
                                        index = self.dialog_act_to_idx[act.lower() + "_" + slot[0].lower()]
                                        current_state[index + len(self.belief_state_to_idx)] = 1
                                    index = self.dialog_act_to_idx[act.lower()]
                                    current_state[index + len(self.belief_state_to_idx)] = 1
                            start, end = str(previous_state), str(current_state)
                            ## Check if this edge already exists in the graph
                            if self.graph.has_edge(start, end):
                                self.graph[start][end]['turn'].append(turn)
                                self.graph[start][end]['probability'] += 1.0
                                repetitive_turns += 1
                            else:
                                self.graph.add_edge(start, end, turn=[turn], probability=1.0)
                                unique_turns += 1
                            ## Update the previous_state for the next turn
                            previous_state = current_state
                            last_belief_state = [0] * len(self.belief_state_to_idx)
                    previous_state = str(previous_state)
                    ## Add a final_state if not exist in the graph
                    if self.graph.has_edge(previous_state, self.final_state):
                        self.graph[previous_state][self.final_state]['probability'] += 1
                    else:
                        self.graph.add_edge(previous_state, self.final_state, turn=[], probability=1.0)
        # nx.write_edgelist(self.graph, "graph.csv", delimiter=';', encoding="utf-8")
        degree = []
        # noinspection PyCallingNonCallable
        for node, value in self.graph.out_degree():
            if value < 21:  # Removing outliers that can really skew the average
                degree.append(value)
        # n, bins, patches = plt.hist(degree, 20, density=True, facecolor='b')
        # plt.xlabel('# of Outgoing Edges (capped at 20 for clearer comparison)')
        # plt.ylabel('% of Outgoing Edges')
        # plt.title('Histogram of Outgoing Edges - MultiWOZ')
        # plt.grid(True)
        # plt.show()

        print("-----------------------------------------------")
        print("Stats for ConvGraph for %s%s" % (self.dir_name, " and ".join(self.file_names)))
        print("Average degree: %2.3f (excluding outliers)" % numpy.mean(degree))
        print("Number of nodes: %d" % len(self.graph.nodes()))
        print("Number of edges: %d" % len(self.graph.edges()))
        print("Number of conversations: %d" % no_of_dialogues)
        print("Unique turns: %d" % unique_turns)
        print("Total turns: %d" % int(unique_turns + repetitive_turns))
        print("As a percentage: %2.3f" % (100 * unique_turns / (unique_turns + repetitive_turns)))
        print("-----------------------------------------------")
        # ------------------------- Calculate Probabilities --------------------------
        for start in self.graph:
            probabilities = []
            for end in self.graph[start]:
                probabilities.append(self.graph[start][end]['probability'])
            for end in self.graph[start]:
                self.graph[start][end]['probability'] = self.graph[start][end]['probability'] / sum(probabilities)

    def _init_graph_vectors(self) -> List[Dict[str, int]]:
        state_vector = set()
        dialog_act_vector = set()
        files = [f for f in listdir(self.dir_name) if isfile(join(self.dir_name, f))
                 and f.endswith(self.file_names[0].split(".")[-1])]
        for input_file in files:
            with open(self.dir_name + input_file, 'r') as f:
                data = json.load(f)
                for dialogue in data:
                    dialogue = data[dialogue]
                    for turn in dialogue['log']:
                        if len(turn['metadata']) > 0:
                            for domain in turn['metadata']:
                                for slot in turn['metadata'][domain]['semi']:
                                    if turn['metadata'][domain]['semi'][slot] not in ["", "not mentioned", "none"]:
                                        state_vector.add(domain + "_" + slot.lower())
                                for slot in turn['metadata'][domain]['book']:
                                    if slot == "booked":
                                        for item in turn['metadata'][domain]['book'][slot]:
                                            for key in item:
                                                state_vector.add(domain + "_" + slot + "_" + key.lower())
                                    else:
                                        if turn['metadata'][domain]['book'][slot] not in ["", "not mentioned", "none"]:
                                            state_vector.add(domain + "_" + slot.lower())
                        if len(turn['dialog_act']) == 0:
                            dialog_act_vector.add("empty_dialogue_act")
                        else:
                            for act in turn['dialog_act']:
                                for slot in turn['dialog_act'][act]:
                                    dialog_act_vector.add(act.lower() + "_" + slot[0].lower())
                                dialog_act_vector.add(act.lower())
        state_vector = list(state_vector)
        state_vector.sort()
        dialog_act_vector = list(dialog_act_vector)
        dialog_act_vector.sort()
        # print("State Vector:", state_vector)
        # print("Dialog_act Vector:", dialog_act_vector)
        return [dict([(s, i) for i, s in enumerate(state_vector)]),
                dict([(s, i) for i, s in enumerate(dialog_act_vector)])]

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
                ## Comment the line below to run on the full dataset
#                 data = {k: data[k] for k in list(data)[:10]}
                for dialogue in data:
                    dialogue = data[dialogue]
                    visited_nodes = [[0] * (len(self.belief_state_to_idx) + len(self.dialog_act_to_idx))]
                    last_belief_state = [0] * len(self.belief_state_to_idx)
                    for turn in dialogue['log']:
                        if len(turn['metadata']) > 0:
                            current_state = [0] * (len(self.belief_state_to_idx) + len(self.dialog_act_to_idx))
                            for domain in turn['metadata']:
                                for slot in turn['metadata'][domain]['semi']:
                                    if turn['metadata'][domain]['semi'][slot] not in ["", "not mentioned", "none"]:
                                        index = self.belief_state_to_idx[domain + "_" + slot.lower()]
                                        current_state[index] = 1
                                        last_belief_state[index] = 1
                                for slot in turn['metadata'][domain]['book']:
                                    if slot == "booked":
                                        for item in turn['metadata'][domain]['book'][slot]:
                                            for key in item:
                                                index = self.belief_state_to_idx[
                                                    domain + "_" + slot + "_" + key.lower()]
                                                current_state[index] = 1
                                                last_belief_state[index] = 1
                                    else:
                                        if turn['metadata'][domain]['book'][slot] not in ["", "not mentioned", "none"]:
                                            index = self.belief_state_to_idx[domain + "_" + slot.lower()]
                                            current_state[index] = 1
                                            last_belief_state[index] = 1
                            if len(turn['dialog_act']) == 0:
                                index = self.dialog_act_to_idx["empty_dialogue_act"]
                                current_state[index + len(self.belief_state_to_idx)] = 1
                            else:
                                for act in turn['dialog_act']:
                                    for slot in turn['dialog_act'][act]:
                                        index = self.dialog_act_to_idx[act.lower() + "_" + slot[0].lower()]
                                        current_state[index + len(self.belief_state_to_idx)] = 1
                                    index = self.dialog_act_to_idx[act.lower()]
                                    current_state[index + len(self.belief_state_to_idx)] = 1
                            visited_nodes.append(current_state)
                        else:
                            current_state = last_belief_state + ([0] * len(self.dialog_act_to_idx))
                            if len(turn['dialog_act']) == 0:
                                index = self.dialog_act_to_idx["empty_dialogue_act"]
                                current_state[index + len(self.belief_state_to_idx)] = 1
                            else:
                                for act in turn['dialog_act']:
                                    for slot in turn['dialog_act'][act]:
                                        index = self.dialog_act_to_idx[act.lower() + "_" + slot[0].lower()]
                                        current_state[index + len(self.belief_state_to_idx)] = 1
                                    index = self.dialog_act_to_idx[act.lower()]
                                    current_state[index + len(self.belief_state_to_idx)] = 1
                            visited_nodes.append(current_state)
                            last_belief_state = [0] * len(self.belief_state_to_idx)
                    visited_nodes.append(eval(self.final_state))
                    # print('number of visited_nodes', len(visited_nodes))

                    assert len(visited_nodes) % 2 == 0
                    if not to_json:
                        for i in range(2, len(visited_nodes) - 1, 2):
                            current_node = visited_nodes[i-1]
                            # current_node = visited_nodes[i]
                            # x = visited_nodes[max(0, i - self.seq_length): i]
                            y = visited_nodes[i][len(self.belief_state_to_idx):]

                            ## Find list of ancestors of the current node with given depth
                            neighbour_list = list(nx.dfs_predecessors(G_reverse, str(current_node), depth_limit=self.net_depth))
                            # print(neighbour_list)
                            ## Add the current_node to neighbour_list and change string to list
                            nei_extend_list = neighbour_list + [str(current_node)]
                            x = list(eval(i) for i in nei_extend_list)

                            x = numpy.array(x, dtype='b')

                            train_x.append(x)
                            train_y.append(y)

                            ## Extract subgraph from G to get subgraph's edge index
                            dialogue_subgraph = G.subgraph(nei_extend_list)

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


        #             else:
        #                 log, last_index = [], 0
        #                 for i in range(2, len(visited_nodes) - 1, 2):
        #                     log.extend(dialogue['log'][last_index: i - 1])
        #                     dialog_acts = [(self.graph[str(visited_nodes[i - 1])][t], t) for t in self.graph[str(visited_nodes[i - 1])]]
        #                     dialog_acts.sort(key=lambda t: t[0]['probability'], reverse=True)
        #                     log.append(dialog_acts[0][0]['turn'][0])
        #                     last_index = i
        #                 self.dialogue_id += 1
        #                 self.augmented_conversations[str(self.dialogue_id)] = {'goal': dict(), 'log': log}
        # if to_json:
        #     with open(self.dir_name + self.file_names[0], "r") as inp:
        #         original_data = json.load(inp)
        #         self.augmented_conversations.update(original_data)
        #         with open(self.dir_name + "output/" + self.file_names[0], "w") as fp:
        #             json.dump(self.augmented_conversations, fp, indent=2, sort_keys=True)

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
        self.augmented_paths.clear()
        train_x, train_y = [], []
        for f_name in self.file_names:
            with open(self.dir_name + f_name, 'r') as f:
                data = json.load(f)
                for dialogue in data:
                    dialogue = data[dialogue]
                    visited_nodes = [[0] * (len(self.belief_state_to_idx) + len(self.dialog_act_to_idx))]
                    last_belief_state = [0] * len(self.belief_state_to_idx)
                    for turn in dialogue['log']:
                        if len(turn['metadata']) > 0:
                            current_state = [0] * (len(self.belief_state_to_idx) + len(self.dialog_act_to_idx))
                            for domain in turn['metadata']:
                                for slot in turn['metadata'][domain]['semi']:
                                    if turn['metadata'][domain]['semi'][slot] not in ["", "not mentioned", "none"]:
                                        index = self.belief_state_to_idx[domain + "_" + slot.lower()]
                                        current_state[index] = 1
                                        last_belief_state[index] = 1
                                for slot in turn['metadata'][domain]['book']:
                                    if slot == "booked":
                                        for item in turn['metadata'][domain]['book'][slot]:
                                            for key in item:
                                                index = self.belief_state_to_idx[
                                                    domain + "_" + slot + "_" + key.lower()]
                                                current_state[index] = 1
                                                last_belief_state[index] = 1
                                    else:
                                        if turn['metadata'][domain]['book'][slot] not in ["", "not mentioned", "none"]:
                                            index = self.belief_state_to_idx[domain + "_" + slot.lower()]
                                            current_state[index] = 1
                                            last_belief_state[index] = 1
                            if len(turn['dialog_act']) == 0:
                                index = self.dialog_act_to_idx["empty_dialogue_act"]
                                current_state[index + len(self.belief_state_to_idx)] = 1
                            else:
                                for act in turn['dialog_act']:
                                    for slot in turn['dialog_act'][act]:
                                        index = self.dialog_act_to_idx[act.lower() + "_" + slot[0].lower()]
                                        current_state[index + len(self.belief_state_to_idx)] = 1
                                    index = self.dialog_act_to_idx[act.lower()]
                                    current_state[index + len(self.belief_state_to_idx)] = 1
                            visited_nodes.append(current_state)
                        else:
                            current_state = last_belief_state + ([0] * len(self.dialog_act_to_idx))
                            if len(turn['dialog_act']) == 0:
                                index = self.dialog_act_to_idx["empty_dialogue_act"]
                                current_state[index + len(self.belief_state_to_idx)] = 1
                            else:
                                for act in turn['dialog_act']:
                                    for slot in turn['dialog_act'][act]:
                                        index = self.dialog_act_to_idx[act.lower() + "_" + slot[0].lower()]
                                        current_state[index + len(self.belief_state_to_idx)] = 1
                                    index = self.dialog_act_to_idx[act.lower()]
                                    current_state[index + len(self.belief_state_to_idx)] = 1
                            visited_nodes.append(current_state)
                            last_belief_state = [0] * len(self.belief_state_to_idx)
                    visited_nodes.append(eval(self.final_state))
                    assert len(visited_nodes) % 2 == 0
                    if not to_json:
                        for i in range(2, len(visited_nodes) - 1, 2):
                            x = visited_nodes[max(0, i - self.seq_length): i]
                            while len(x) < self.seq_length:
                                x.insert(0, [0] * (len(self.belief_state_to_idx) + len(self.dialog_act_to_idx)))
                            dialog_acts = [(self.graph[str(visited_nodes[i - 1])][t], t) for t in
                                           self.graph[str(visited_nodes[i - 1])]]
                            dialog_acts.sort(key=lambda t: t[0]['probability'], reverse=True)
                            for dialog_act in dialog_acts[:1]:
                                if dialog_act[1] == self.final_state:
                                    continue
                                y = eval(dialog_act[1])[len(self.belief_state_to_idx):]
                                if str(x) + str(y) not in self.augmented_paths:
                                    train_x.append(x)
                                    train_y.append(y)
                                    self.augmented_paths.add(str(x) + str(y))
                    else:
                        log, last_index = [], 0
                        for i in range(2, len(visited_nodes) - 1, 2):
                            log.extend(dialogue['log'][last_index: i - 1])
                            dialog_acts = [(self.graph[str(visited_nodes[i - 1])][t], t) for t in
                                           self.graph[str(visited_nodes[i - 1])]]
                            dialog_acts.sort(key=lambda t: t[0]['probability'], reverse=True)
                            log.append(dialog_acts[0][0]['turn'][0])
                            last_index = i
                        self.dialogue_id += 1
                        self.augmented_conversations[str(self.dialogue_id)] = {'goal': dict(), 'log': log}
        if to_json:
            with open(self.dir_name + self.file_names[0], "r") as inp:
                original_data = json.load(inp)
                self.augmented_conversations.update(original_data)
                with open(self.dir_name + "output/" + self.file_names[0], "w") as fp:
                    json.dump(self.augmented_conversations, fp, indent=2, sort_keys=True)
        return numpy.array(train_x, dtype='float32'), numpy.array(train_y, dtype='float32')

    def generate_standard_data(self, unique: bool) -> numpy.array:
        self.augmented_paths.clear()
        train_x, train_y = [], []
        for f_name in self.file_names:
            with open(self.dir_name + f_name, 'r') as f:
                data = json.load(f)
                data = {k: data[k] for k in list(data)[:10]}
                for dialogue in data:
                    dialogue = data[dialogue]
                    visited_nodes = [[0] * (len(self.belief_state_to_idx) + len(self.dialog_act_to_idx))]
                    last_belief_state = [0] * len(self.belief_state_to_idx)
                    for turn in dialogue['log']:
                        if len(turn['metadata']) > 0:
                            current_state = [0] * (len(self.belief_state_to_idx) + len(self.dialog_act_to_idx))
                            for domain in turn['metadata']:
                                for slot in turn['metadata'][domain]['semi']:
                                    if turn['metadata'][domain]['semi'][slot] not in ["", "not mentioned", "none"]:
                                        index = self.belief_state_to_idx[domain + "_" + slot.lower()]
                                        current_state[index] = 1
                                        last_belief_state[index] = 1
                                for slot in turn['metadata'][domain]['book']:
                                    if slot == "booked":
                                        for item in turn['metadata'][domain]['book'][slot]:
                                            for key in item:
                                                index = self.belief_state_to_idx[
                                                    domain + "_" + slot + "_" + key.lower()]
                                                current_state[index] = 1
                                                last_belief_state[index] = 1
                                    else:
                                        if turn['metadata'][domain]['book'][slot] not in ["", "not mentioned", "none"]:
                                            index = self.belief_state_to_idx[domain + "_" + slot.lower()]
                                            current_state[index] = 1
                                            last_belief_state[index] = 1
                            if len(turn['dialog_act']) == 0:
                                index = self.dialog_act_to_idx["empty_dialogue_act"]
                                current_state[index + len(self.belief_state_to_idx)] = 1
                            else:
                                for act in turn['dialog_act']:
                                    for slot in turn['dialog_act'][act]:
                                        index = self.dialog_act_to_idx[act.lower() + "_" + slot[0].lower()]
                                        current_state[index + len(self.belief_state_to_idx)] = 1
                                    index = self.dialog_act_to_idx[act.lower()]
                                    current_state[index + len(self.belief_state_to_idx)] = 1
                            visited_nodes.append(current_state)
                        else:
                            current_state = last_belief_state + ([0] * len(self.dialog_act_to_idx))
                            if len(turn['dialog_act']) == 0:
                                index = self.dialog_act_to_idx["empty_dialogue_act"]
                                current_state[index + len(self.belief_state_to_idx)] = 1
                            else:
                                for act in turn['dialog_act']:
                                    for slot in turn['dialog_act'][act]:
                                        index = self.dialog_act_to_idx[act.lower() + "_" + slot[0].lower()]
                                        current_state[index + len(self.belief_state_to_idx)] = 1
                                    index = self.dialog_act_to_idx[act.lower()]
                                    current_state[index + len(self.belief_state_to_idx)] = 1
                            visited_nodes.append(current_state)
                            last_belief_state = [0] * len(self.belief_state_to_idx)
                    visited_nodes.append(eval(self.final_state))
                    assert len(visited_nodes) % 2 == 0
                    for i in range(2, len(visited_nodes) - 1, 2):
                        x = visited_nodes[max(0, i - self.seq_length): i]
                        y = visited_nodes[i][len(self.belief_state_to_idx):]
                        while len(x) < self.seq_length:
                            x.insert(0, [0] * (len(self.belief_state_to_idx) + len(self.dialog_act_to_idx)))
                        if unique:
                            if str(x) + str(y) not in self.augmented_paths:
                                train_x.append(x)
                                train_y.append(y)
                                self.augmented_paths.add(str(x) + str(y))
                        else:
                            train_x.append(x)
                            train_y.append(y)
        return numpy.array(train_x, dtype='float32'), numpy.array(train_y, dtype='float32')

    def get_valid_dialog_acts(self, state: List[List[float]]):
        current_state = str([int(value) for value in state[-1]])
        dialog_acts = [eval(node)[-len(self.dialog_act_to_idx):] for node in self.graph[current_state]]
        return dialog_acts

    def get_best_f1_score(self, state: List[List[float]], y_pred: List[float]) -> Tuple[List[int], float]:
        dialog_acts = self.get_valid_dialog_acts(state)
        f1_scores = []
        for y_true in dialog_acts:
            f1_scores.append(f1_score(y_pred=y_pred, y_true=y_true))
        return dialog_acts[numpy.argmax(f1_scores)], max(f1_scores)


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

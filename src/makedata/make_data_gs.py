import networkx as nx
import numpy as np
from collections import defaultdict

def load_data(parent="./data/raw/", data="cora", features="cora.content", edges="cora.cites"):
    # calculate num nodes and features size using networkx
    par_path = parent+data
    G = nx.Graph()
    class_label = set()
    with open(f"{par_path}/{edges}") as f:
        for line in f.readlines():
            node1, node2 = line.split()
            G.add_edge(node1, node2)
    num_nodes = G.number_of_nodes()
    with open(f"{par_path}/{features}") as f:
        all_text = f.readlines()
        num_feats = len(all_text[0].split()[1:-1])
        for line in all_text:
            label = line.strip().split()[-1]
            class_label.add(label)
    num_label = len(class_label)

    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}

    with open(f"{par_path}/{features}") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            # print(info[1:-1])
            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]
    # print(f"finished skinning {par_path}/{features}")

    adj_lists = defaultdict(set)
    with open(f"{par_path}/{edges}") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    # print(f"finished skinning {par_path}/{edges}")

    return feat_data, labels, adj_lists, num_nodes, num_feats, num_label
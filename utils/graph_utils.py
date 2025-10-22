
import json
import networkx as nx
import numpy as np
import pdb

from . import process_TEP_graph, process_SWaT_graph, process_WADI_graph, process_CTOWN_graph
from . import attack_utils, tep_utils

def convert_attention_to_graph(learned_graph):

    np_graph = np.array(learned_graph)
    edgelist = []
    
    # learned_graph is a (D,K) shape of dsts for each src row
    for row_index in range(len(np_graph)):
        learned_row = np_graph[row_index]
        for j in learned_row:
            edgelist.append((row_index, j))

    graph = nx.from_edgelist(edgelist)
    return graph

def convert_edge_indexs_to_graph(edge_indexs):

    np_edges = np.array(edge_indexs)
    edgelist = []

    # edge_indexes is a (2,E) input for message passing.
    for i in range(np_edges.shape[1]):
        edgelist.append((np_edges[0][i], np_edges[1][i]))

    graph = nx.from_edgelist(edgelist)
    return graph

def get_fc_graph_struc(feature_list):
    
    struc_map = {}

    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []

        for other_ft in feature_list:
            if other_ft is not ft:
                struc_map[ft].append(other_ft)
    
    return struc_map

def build_fc_net(n_features):

    edge_indexes = [[],[]]

    for i in range(n_features):
        
        for j in range(n_features):

            if i == j:
                continue

            edge_indexes[0].append(i)
            edge_indexes[1].append(j)
    
    return edge_indexes

def build_loc_net(struc, feature_list, feature_map=[]):

    index_feature_map = feature_map
    edge_indexes = [
        [],
        []
    ]

    for node_name, node_list in struc.items():
        if node_name not in feature_list:
            continue

        if node_name not in index_feature_map:
            index_feature_map.append(node_name)
        
        p_index = index_feature_map.index(node_name)
        for child in node_list:
            if child not in feature_list:
                continue

            if child not in index_feature_map:
                print(f'error: {child} not in index_feature_map')
                # index_feature_map.append(child)

            c_index = index_feature_map.index(child)
            # edge_indexes[0].append(p_index)
            # edge_indexes[1].append(c_index)
            edge_indexes[0].append(c_index)
            edge_indexes[1].append(p_index)

    return edge_indexes

# A mock graph for testing: each node is connected to the next two nodes in order.
def build_neighbor_net(struc, feature_list, feature_map=[]):

    edge_indexes = [[],[]]

    for idx in range(len(feature_list)):
        
        edge_indexes[0].append(idx)
        edge_indexes[1].append((idx + 1) % len(feature_list))

        edge_indexes[0].append(idx)
        edge_indexes[1].append((idx + 2) % len(feature_list))
    
    return edge_indexes

def build_full_spec_graph_idxs(dataset, sensor_cols=None):

    all_srcs = []
    all_dsts = []
    
    if dataset == 'TEP':
        all_edges = process_TEP_graph.PROC_EDGES + process_TEP_graph.PID_EDGES
    elif dataset == 'CTOWN':
        all_edges = process_CTOWN_graph.PROC_EDGES
    elif dataset == 'SWAT':
        all_edges = process_SWaT_graph.PROC_EDGES
    elif dataset == 'WADI':
        all_edges = process_WADI_graph.PROC_EDGES
    else:
        print(f'Dataset "{dataset}" not found')

    for i in range(len(all_edges)):

        src = all_edges[i][0]
        dst = all_edges[i][1]
      
        if dataset == "TEP":
            sidx = tep_utils.sen_to_idx(src)
            didx = tep_utils.sen_to_idx(dst)
        elif dataset == "CTOWN":
            sidx = process_CTOWN_graph.sen_to_idx(src, sensor_cols)
            didx = process_CTOWN_graph.sen_to_idx(dst, sensor_cols)
        elif dataset == "SWAT":
            sidx = process_SWaT_graph.sen_to_idx(src, sensor_cols)
            didx = process_SWaT_graph.sen_to_idx(dst, sensor_cols)
        elif dataset == "WADI":
            sidx = process_WADI_graph.sen_to_idx(src, sensor_cols)
            didx = process_WADI_graph.sen_to_idx(dst, sensor_cols)

        #print(f'Adding: {sidx} {didx}')
        all_srcs.append(sidx)
        all_dsts.append(didx)

    return all_srcs, all_dsts

def build_grouped_graph_idxs(dataset, subprocess=False):

    all_srcs = []
    all_dsts = []
    
    if dataset == 'TEP':

        if subprocess:
            sub_idxs, sub_labels = process_TEP_graph.get_sensor_subsets_name()
        else:
            sub_idxs, sub_labels = process_TEP_graph.get_sensor_subsets_simulink()
    
    elif dataset == 'SWAT':
        
        if subprocess:
            sub_idxs, sub_labels = process_SWaT_graph.get_sensor_subsets_subprocesses()
        else:
            sub_idxs, sub_labels = process_SWaT_graph.get_sensor_subsets_plcs()

    elif dataset == 'CTOWN':
        sub_idxs, sub_labels = process_CTOWN_graph.get_sensor_subsets()

    elif dataset == 'WADI':
        sub_idxs, sub_labels = process_WADI_graph.get_sensor_subsets()
    else:
        print(f'Dataset "{dataset}" not found')

    for si in range(len(sub_idxs)):

        for src in sub_idxs[si]:
            for dst in sub_idxs[si]:

                if src == dst:
                    continue

                all_srcs.append(src)
                all_dsts.append(dst)

    return all_srcs, all_dsts

def build_graph_from_spec_file(dataset, filename, sensor_cols=None):

    with open(filename, 'r') as f:
        data = json.load(f)

    assert dataset in data['dataset']

    all_edges = data['edges']
    all_srcs = []
    all_dsts = []
    
    for i in range(len(all_edges)):

        src = all_edges[i][0]
        dst = all_edges[i][1]
      
        if dataset == "TEP":
            sidx = tep_utils.sen_to_idx(src)
            didx = tep_utils.sen_to_idx(dst)
        elif dataset == "CTOWN":
            sidx = process_CTOWN_graph.sen_to_idx(src, sensor_cols)
            didx = process_CTOWN_graph.sen_to_idx(dst, sensor_cols)
        elif dataset == "SWAT":
            sidx = process_SWaT_graph.sen_to_idx(src, sensor_cols)
            didx = process_SWaT_graph.sen_to_idx(dst, sensor_cols)
        elif dataset == "WADI":
            sidx = process_WADI_graph.sen_to_idx(src, sensor_cols)
            didx = process_WADI_graph.sen_to_idx(dst, sensor_cols)

        #print(f'Adding: {sidx} {didx}')
        all_srcs.append(sidx)
        all_dsts.append(didx)

    return all_srcs, all_dsts


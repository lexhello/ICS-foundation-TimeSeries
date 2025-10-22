import torch
from torch_geometric.utils import degree
import numpy as np
import pdb

def compute_edge_norm(edge_index, num_nodes):
    
    # Compute normalization.
    
    row, col = edge_index
    deg = degree(col, num_nodes, dtype=torch.float32)

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
    return norm

def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()

def convert_to_time_edge_index(org_edge_index, time_window, node_num):

    # Assumes the standard [2, E] edge_index size, where E corresponds to sensor idxs
    structure_start_set = np.array(org_edge_index[0].clone().cpu())
    structure_end_set = np.array(org_edge_index[1].clone().cpu())

    # We want to duplicate the original edges to include time, stored here
    new_start_edges = np.array([])
    new_end_edges = np.array([])

    # Build time based structure across rows: a_t-1 -> a_t for all features
    for nn in range(node_num):
        
        new_time_start_edges = np.arange(nn * time_window, (nn + 1) * time_window - 1)
        new_time_end_edges = np.arange(nn * time_window + 1, (nn + 1) * time_window)
        
        new_start_edges = np.concatenate([new_start_edges, new_time_start_edges])
        new_end_edges = np.concatenate([new_end_edges, new_time_end_edges])

    # Copy dependencies across time: a -> b copies to a_t -> b_t+1 for all t
    for t in range(time_window - 1):
        
        new_structure_start_edges = (structure_start_set * time_window) + t
        new_structure_end_edges = (structure_end_set * time_window) + t + 1

        new_start_edges = np.concatenate([new_start_edges, new_structure_start_edges])
        new_end_edges = np.concatenate([new_end_edges, new_structure_end_edges])

    new_edge_index = np.vstack([new_start_edges, new_end_edges])
    
    return new_edge_index

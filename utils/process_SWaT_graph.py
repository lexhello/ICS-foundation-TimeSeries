import pdb
import pandas as pd
import numpy as np
import networkx as nx

from . import attack_utils
from data_loader import load_train_data

PROC_EDGES = [
   
   # Raw water tank
   ('MV101', 'FIT101'),
   ('FIT101', 'LIT101'),
   
   # Pump from raw water to dosing
   ('P101', 'FIT201'),
   ('P102', 'FIT201'),

   # Dosing pump control
   ('FIT201', 'P201'),
   ('FIT201', 'P203'),
   ('FIT201', 'P205'),

   # Dosing pump outputs
   #('P201', 'AIT201'),
   ('P203', 'AIT202'),
   ('P205', 'AIT203'),

   # UF Feed
   ('MV201', 'LIT301'),
   ('MV301', 'DPIT301'),
   ('MV302', 'DPIT301'),
   
   # UF out
   ('P301', 'FIT301'),
   ('P302', 'FIT301'),
   ('FIT301', 'DPIT301'),

   # RO Feed in
   ('MV301', 'LIT401'),
   ('MV302', 'LIT401'),
   ('MV303', 'LIT401'),
   ('MV304', 'LIT401'),

   # UV in
   #('P401', 'FIT401'),
   ('P402', 'FIT401'),
   ('FIT401', 'UV401'),
   ('AIT402', 'P203'),
   ('AIT402', 'P205'),

   # What causes AIT401/402?
   # What does P403 do?
   ('P501', 'PIT501'),
   #('P502', 'PIT501'),
   ('FIT501', 'PIT501'),
   ('FIT504', 'PIT501'),
   ('FIT502', 'PIT502'),
   ('FIT503', 'PIT503'),

   # Backwash pumps
   #('P601', 'LIT101'),
   ('P602', 'FIT601'),
   ('FIT601', 'LIT401'),
   ('DPIT301', 'P602')
]

def get_sensor_subsets_plcs():

   subprocess_idxs = [
      [0, 1, 2, 3],                   # PLC 1
      [5, 6, 7, 8, 9, 10, 11, 12, 13, 14], # PLC 2 without AIT201 (originally feature #5)
      [8, 15, 16, 17, 18, 19, 22],    # PLC 3 (downshifted by 1)
      [25, 26, 27, 28],               # PLC 4 (downshifted by 1)
      [33, 37, 38, 41],               # PLC 5 (downshifted by 1)
      #[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], # PLC 2
      # [9, 16, 17, 18, 19, 20, 23],    # PLC 3
      # [26, 27, 28, 29],               # PLC 4
      # [34, 38, 39, 42],               # PLC 5
   ]

   subprocess_labels =[
      ['FIT101', 'LIT101', 'MV101', 'P101'],
      #['AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206'],
      ['AIT202', 'AIT203', 'FIT201', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206'],
      ['MV201', 'DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'P301'],
      ['AIT402', 'FIT401', 'LIT401', 'P401'],
      ['AIT501', 'FIT501', 'FIT502', 'P501'],
   ]

   return subprocess_idxs, subprocess_labels

def get_sensor_subsets_subprocesses():

   SWAT_SUB_MAP = {
      '1_Raw_Water_Tank' : ['MV101', 'LIT101', 'FIT101', 'P101', 'P102'],
      #'2_Chemical' : ['P201', 'P202', 'P203', 'P204', 'P205', 'P206', 'FIT201', 'AIT201', 'AIT202', 'AIT203', 'MV201'], 
      '2_Chemical' : ['P201', 'P202', 'P203', 'P204', 'P205', 'P206', 'FIT201', 'AIT202', 'AIT203', 'MV201'], # if AIT201 causes too much bias
      '3_UltraFilt' : ['FIT301', 'LIT301', 'DPIT301', 'P301', 'P302', 'MV301', 'MV302', 'MV303', 'MV304'],
      '4_DeChloro' : ['UV401', 'P401', 'P402', 'P403', 'P404', 'AIT401', 'AIT402', 'FIT401', 'LIT401'],
      '5_RO' : ['AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'P502', 'PIT501', 'PIT502', 'PIT503'],
      '6_Return' : ['P601', 'P602', 'P603', 'FIT601']
   }

   subprocess_idxs = [
      [0, 1, 2, 3, 4],                   # 100s
      [5, 6, 7, 8, 9, 10, 11, 12, 13, 14], # 200s
      [15, 16, 17, 18, 19, 20, 21, 22, 23],    # 300s
      [24, 25, 26, 27, 28, 29, 30, 31, 32],    # 400s
      [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45],  # 500s
      [46, 47, 48, 49],  # 500s
      # [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], # 200s
      # [16, 17, 18, 19, 20, 21, 22, 23, 24],    # 300s
      # [25, 26, 27, 28, 29, 30, 31, 32, 33],    # 400s
      # [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46],  # 500s
      # [47, 48, 49, 50],  # 500s
   ]
   subprocess_labels =[
      ['MV101', 'LIT101', 'FIT101', 'P101', 'P102'],
      #['P201', 'P202', 'P203', 'P204', 'P205', 'P206', 'FIT201', 'AIT201', 'AIT202', 'AIT203', 'MV201'], 
      ['P201', 'P202', 'P203', 'P204', 'P205', 'P206', 'FIT201', 'AIT202', 'AIT203', 'MV201'], 
      ['FIT301', 'LIT301', 'DPIT301', 'P301', 'P302', 'MV301', 'MV302', 'MV303', 'MV304'],
      ['UV401', 'P401', 'P402', 'P403', 'P404', 'AIT401', 'AIT402', 'FIT401', 'LIT401'],
      ['AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'P502', 'PIT501', 'PIT502', 'PIT503'],
      ['P601', 'P602', 'P603', 'FIT601']
   ]

   return subprocess_idxs, subprocess_labels

def idx_to_sen(idx, sensor_cols):
   return sensor_cols[idx]

def sen_to_idx(sensor, sensor_cols):
   return sensor_cols.index(sensor)

def get_subprocess_idx(sensor, subprocess_labels):

   for i in range(len(subprocess_labels)):
      
      subprocess = subprocess_labels[i]
      if sensor in subprocess:
         return i

   return -1

def build_full_swat_graph():

   dataset_name = 'SWAT'

   Xtrain, sensor_cols = load_train_data(dataset_name)
   cov = np.cov(Xtrain[::10].T)

   ###############################
   ### Build Directed Graph
   ###############################
   G = nx.DiGraph()
   G.add_nodes_from(sensor_cols)

   # Add all pairwise edges
   for sidx in range(len(sensor_cols)):
      for didx in range(len(sensor_cols)):

         if sidx == didx:
            continue

         src = sensor_cols[sidx]
         dst = sensor_cols[didx]

         if dataset_name == 'SWAT-PHY':   
            if 'AIT' in src or 'AIT' in dst:
               print(f'Skipping {src} {dst}')
               continue

         G.add_edge(src, dst, weight=np.abs(cov[sidx][didx]))

   nx.write_gml(G, f"explanations-dir/graph-{dataset_name}-full.gml")

def build_proc_swat_graph():

   dataset_name = 'SWAT'

   Xtrain, sensor_cols = load_train_data(dataset_name)
   cov = np.cov(Xtrain[::10].T)

   ###############################
   ### Build Directed Graph
   ###############################
   G = nx.DiGraph()
   G.add_nodes_from(sensor_cols)

   for i in range(len(PROC_EDGES)):

      src = PROC_EDGES[i][0]
      dst = PROC_EDGES[i][1]
      
      sidx = sen_to_idx(src, sensor_cols)
      didx = sen_to_idx(dst, sensor_cols)

      if dataset_name == 'SWAT-PHY':
         if 'AIT' in src or 'AIT' in dst:
            print(f'Skipping {src} {dst}')
            continue

      if np.abs(cov[sidx][didx]) == 0:
         print(f'Note: a 0-covariance edge is being added: {src} {dst}')

      #G.add_edge(src, src, weight=np.abs(cov[sidx][sidx]))
      #G.add_edge(dst, src, weight=np.abs(cov[sidx][didx]))
      G.add_edge(src, dst, weight=np.abs(cov[sidx][didx]))

   UG = G.to_undirected()
   nx.write_gml(G, f"explanations-dir/graph-{dataset_name}-proc.gml")
   nx.write_gml(UG, f"explanations-dir/graph-ud-{dataset_name}-proc.gml")

def build_plc_swat_graph():

   sub_idxs, sub_labels = attack_utils.get_sensor_subsets('SWAT')

   dataset_name = 'SWAT'
   Xtrain_clean, sensor_cols_clean = load_train_data(dataset_name)

   cov = np.cov(Xtrain_clean[::10].T)

   ###############################
   ### Build Directed Graph
   ###############################
   G = nx.DiGraph()
   G.add_nodes_from(sensor_cols_clean)

   for i in range(len(sensor_cols_clean)):
      for j in range(len(sensor_cols_clean)):

         if i == j:
            continue

         src = sensor_cols_clean[i]
         dst = sensor_cols_clean[j]
         
         sidx = sen_to_idx(src, sensor_cols_clean)
         didx = sen_to_idx(dst, sensor_cols_clean)

         src_idx = get_subprocess_idx(src, sub_labels)
         dst_idx = get_subprocess_idx(dst, sub_labels)

         if src_idx == -1:
            continue

         elif dst_idx == -1:
            continue

         elif src_idx == dst_idx:
            print(f'{src} and {dst}')
            G.add_edge(src, dst, weight=np.abs(cov[sidx][didx]))
            G.add_edge(dst, src, weight=np.abs(cov[sidx][didx]))

   UG = G.to_undirected()
   nx.write_gml(G, f"explanations-dir/graph-{dataset_name}-plcs.gml")
   nx.write_gml(UG, f"explanations-dir/graph-ud-{dataset_name}-plcs.gml")

if __name__ == '__main__':

   #build_full_swat_graph()
   build_proc_swat_graph()
   build_plc_swat_graph()

import pdb
import pandas as pd
import numpy as np
import networkx as nx

from . import attack_utils
from data_loader import load_train_data, load_test_data

PROC_EDGES = [
	
   # Inputs to Raw water
   ('1_MV_001_STATUS', '1_FIT_001_PV'),
   ('1_MV_004_STATUS', '1_FIT_001_PV'),
   #('3_MV_002_STATUS', '1_FIT_001_PV'),
   ('1_FIT_001_PV', '1_LT_001_PV'),

   # First dosing
   # ('1_P_001_STATUS', '1_LS_001_AL'),
   # ('1_P_002_STATUS', '1_LS_001_AL'),
   # ('1_P_003_STATUS', '1_LS_002_AL'),
   # ('1_P_004_STATUS', '1_LS_002_AL'),

   ('1_AIT_001_PV', '1_LT_001_PV'),
   ('1_AIT_002_PV', '1_LT_001_PV'),
   ('1_AIT_003_PV', '1_LT_001_PV'),
   ('1_AIT_004_PV', '1_LT_001_PV'),
   ('1_AIT_005_PV', '1_LT_001_PV'),

   #('1_MV_003_STATUS', '1_LT_001_PV'),
   ('1_LT_001_PV', '1_P_005_STATUS'),
   ('1_LT_001_PV', '1_P_006_STATUS'),

   # Inputs to Elevated Reservoir
   ('1_P_005_STATUS', '2_FIT_001_PV'),
   ('1_P_006_STATUS', '2_FIT_001_PV'),
   ('2_FIT_001_PV', '2_LT_001_PV'),
   ('2_FIT_001_PV', '2_LT_002_PV'),
   # ('1_P_005_STATUS', '2_LT_001_PV'),
   # ('1_P_006_STATUS', '2_LT_001_PV'),
   # ('1_P_005_STATUS', '2_LT_002_PV'),
   # ('1_P_006_STATUS', '2_LT_002_PV'),
   #('2_MV_001_STATUS', '2_LT_001_PV'),
   #('2_MV_002_STATUS', '2_LT_001_PV'),
   ('2_MV_003_STATUS', '2_LT_002_PV'),
   #('2_MV_004_STATUS', '2_LT_002_PV'),

   # Output of Elevated Reservoir
   # ('2_MV_002_STATUS', '2_PIT_001_PV'),
   # ('2_MV_004_STATUS', '2_PIT_001_PV'),

   # ER chemical flows
   ('2A_AIT_001_PV', '2_PIT_001_PV'),
   ('2A_AIT_002_PV', '2_PIT_001_PV'),
   ('2A_AIT_003_PV', '2_PIT_001_PV'),
   ('2A_AIT_004_PV', '2_PIT_001_PV'),
   ('2_PIT_001_PV', '2A_AIT_001_PV'),
   ('2_PIT_001_PV', '2A_AIT_002_PV'),
   ('2_PIT_001_PV', '2A_AIT_003_PV'),
   ('2_PIT_001_PV', '2A_AIT_004_PV'),

   # Output of 2A
   #('2_MV_005_STATUS', '2_PIT_001_PV'),
   ('2_MV_006_STATUS', '2_PIT_001_PV'),
   #('2_MV_005_STATUS', '2_FIT_002_PV'),

   # Double containment feed
   #('2_MV_009_STATUS', '2_PIT_002_PV'),
   ('2_MCV_007_CO', '2_PIT_002_PV'),
   ('2_MCV_007_CO', '2_FIT_002_PV'),
   ('2B_AIT_001_PV', '2_FIT_002_PV'),
   ('2B_AIT_003_PV', '2_FIT_002_PV'),
   ('2B_AIT_004_PV', '2_FIT_002_PV'),
   ('2_FIT_002_PV', '2B_AIT_001_PV'),
   ('2_FIT_002_PV', '2B_AIT_003_PV'),
   ('2_FIT_002_PV', '2B_AIT_004_PV'),
   
   # Booster flow
   ('2_P_003_STATUS', '2_PIT_003_PV'),
   ('2_P_003_STATUS', '2_FIT_003_PV'),
   #('2_P_004_STATUS', '2_PIT_003_PV'),
   #('2_P_004_STATUS', '2_FIT_003_PV'),
   ('2_PIT_003_PV', '2_FIT_003_PV'),
   ('2_PIC_003_PV', '2_FIT_003_PV'),
   ('2_FIT_003_PV', '2_PIT_003_PV'),
   ('2_PIC_003_PV', '2_PIT_003_PV'),
   ('2_PIT_003_PV', '2_PIC_003_PV'),
   ('2_FIT_003_PV', '2_PIC_003_PV'),

   # Consumer tank flow (unsure about this)
   ('2_PIT_003_PV', '2_MCV_101_CO'),
   ('2_PIT_003_PV', '2_MCV_201_CO'),
   ('2_PIT_003_PV', '2_MCV_301_CO'),
   ('2_PIT_003_PV', '2_MCV_401_CO'),
   ('2_PIT_003_PV', '2_MCV_501_CO'),
   ('2_PIT_003_PV', '2_MCV_601_CO'),
   ('2_FIT_003_PV', '2_MCV_101_CO'),
   ('2_FIT_003_PV', '2_MCV_201_CO'),
   ('2_FIT_003_PV', '2_MCV_301_CO'),
   ('2_FIT_003_PV', '2_MCV_401_CO'),
   ('2_FIT_003_PV', '2_MCV_501_CO'),
   ('2_FIT_003_PV', '2_MCV_601_CO'),

   # Per Consumer Tank (FIC-CO? FIC-PV, FIC-SP)
   ('2_FIC_101_CO', '2_FQ_101_PV'),
   ('2_FIC_101_CO', '2_FIC_101_PV'),
   ('2_FIC_101_CO', '2_FIC_101_SP'),
   ('2_MCV_101_CO', '2_LS_101_AH'),
   ('2_MCV_101_CO', '2_LS_101_AL'),
   ('2_MV_101_STATUS', '2_LS_101_AH'),
   ('2_MV_101_STATUS', '2_LS_101_AL'),
   # ('2_SV_101_STATUS', '2_LS_101_AH'),
   # ('2_SV_101_STATUS', '2_LS_101_AL'),

   ('2_FIC_201_CO', '2_FQ_201_PV'),
   ('2_FIC_201_CO', '2_FIC_201_PV'),
   ('2_FIC_201_CO', '2_FIC_201_SP'),
   ('2_MCV_201_CO', '2_LS_201_AH'),
   ('2_MCV_201_CO', '2_LS_201_AL'),
   ('2_MV_201_STATUS', '2_LS_201_AH'),
   ('2_MV_201_STATUS', '2_LS_201_AL'),
   # ('2_SV_201_STATUS', '2_LS_201_AH'),
   # ('2_SV_201_STATUS', '2_LS_201_AL'),

   ('2_FIC_301_CO', '2_FQ_301_PV'),
   ('2_FIC_301_CO', '2_FIC_301_PV'),
   ('2_FIC_301_CO', '2_FIC_301_SP'),
   ('2_MCV_301_CO', '2_LS_301_AH'),
   ('2_MCV_301_CO', '2_LS_301_AL'),
   ('2_MV_301_STATUS', '2_LS_301_AH'),
   ('2_MV_301_STATUS', '2_LS_301_AL'),
   # ('2_SV_301_STATUS', '2_LS_301_AH'),
   # ('2_SV_301_STATUS', '2_LS_301_AL'),

   ('2_FIC_401_CO', '2_FQ_401_PV'),
   ('2_FIC_401_CO', '2_FIC_401_PV'),
   ('2_FIC_401_CO', '2_FIC_401_SP'),
   ('2_MCV_401_CO', '2_LS_401_AH'),
   ('2_MCV_401_CO', '2_LS_401_AL'),
   ('2_MV_401_STATUS', '2_LS_401_AH'),
   ('2_MV_401_STATUS', '2_LS_401_AL'),
   # ('2_SV_401_STATUS', '2_LS_401_AH'),
   # ('2_SV_401_STATUS', '2_LS_401_AL'),

   ('2_FIC_501_CO', '2_FQ_501_PV'),
   ('2_FIC_501_CO', '2_FIC_501_PV'),
   ('2_FIC_501_CO', '2_FIC_501_SP'),
   ('2_MCV_501_CO', '2_LS_501_AH'),
   ('2_MCV_501_CO', '2_LS_501_AL'),
   ('2_MV_501_STATUS', '2_LS_501_AH'),
   ('2_MV_501_STATUS', '2_LS_501_AL'),
   # ('2_SV_501_STATUS', '2_LS_501_AH'),
   # ('2_SV_501_STATUS', '2_LS_501_AL'),

   ('2_FIC_601_CO', '2_FQ_601_PV'),
   ('2_FIC_601_CO', '2_FIC_601_PV'),
   ('2_FIC_601_CO', '2_FIC_601_SP'),
   ('2_MCV_601_CO', '2_LS_601_AH'),
   ('2_MCV_601_CO', '2_LS_601_AL'),
   ('2_MV_601_STATUS', '2_LS_601_AH'),
   ('2_MV_601_STATUS', '2_LS_601_AL'),
   # ('2_SV_601_STATUS', '2_LS_601_AH'),
   # ('2_SV_601_STATUS', '2_LS_601_AL'),

   # Stage 3 input
   ('2_MV_101_STATUS', '3_FIT_001_PV'),
   ('2_MV_201_STATUS', '3_FIT_001_PV'),
   ('2_MV_301_STATUS', '3_FIT_001_PV'),
   ('2_MV_401_STATUS', '3_FIT_001_PV'),
   ('2_MV_501_STATUS', '3_FIT_001_PV'),
   ('2_MV_601_STATUS', '3_FIT_001_PV'),

   # Stage 3 chemical
   # ('3_P_001_STATUS', '3_LS_001_AL'),
   # ('3_P_002_STATUS', '3_LS_001_AL'),

   # Stage 3 flow
   ('3_FIT_001_PV', '3_LT_001_PV'),
   # ('3_MV_001_STATUS', '3_LT_001_PV'),
   # ('3_MV_002_STATUS', '3_LT_001_PV'),
   # ('3_MV_003_STATUS', '3_LT_001_PV'),
   #('3_P_003_STATUS', '3_LT_001_PV'),
   #('3_P_004_STATUS', '3_LT_001_PV'),

]

def get_sensor_subsets():

   WADI_SUB_MAP = {
	'1_Raw_Water_Tank' : ['1_AIT_001_PV', '1_AIT_002_PV', '1_AIT_003_PV', '1_AIT_004_PV', '1_AIT_005_PV',
		'1_FIT_001_PV', '1_LS_001_AL', '1_LS_002_AL', '1_LT_001_PV',
		'1_MV_001_STATUS', '1_MV_002_STATUS', '1_MV_003_STATUS', '1_MV_004_STATUS',
		'1_P_001_STATUS', '1_P_002_STATUS', '1_P_003_STATUS', '1_P_004_STATUS', '1_P_005_STATUS', '1_P_006_STATUS'],
	'Elevated' : ['2_FIT_001_PV', '2_FIT_002_PV', '2_FIT_003_PV', '2_LT_001_PV', '2_LT_002_PV', '2_PIT_001_PV',
	 	'2_MV_001_STATUS', '2_MV_002_STATUS', '2_MV_003_STATUS', '2_MV_004_STATUS', '2_MV_005_STATUS', '2_MV_006_STATUS',
		'2A_AIT_001_PV', '2A_AIT_002_PV', '2A_AIT_003_PV', '2A_AIT_004_PV',],
	'Booster': ['2_DPIT_001_PV', '2_MCV_007_CO', '2_MV_009_STATUS', 
		'2_P_003_SPEED', '2_P_003_STATUS', '2_P_004_SPEED', '2_P_004_STATUS',
		'2_PIT_002_PV', '2_PIT_003_PV', '2B_AIT_001_PV', '2B_AIT_003_PV', '2B_AIT_004_PV',
		'2_PIC_003_CO', '2_PIC_003_PV', '2_PIC_003_SP'],
	'Consumers': ['2_FIC_101_CO', '2_FIC_101_PV', '2_FIC_101_SP', '2_FIC_201_CO', '2_FIC_201_PV', '2_FIC_201_SP', '2_FIC_301_CO', '2_FIC_301_PV', '2_FIC_301_SP', 
		'2_FIC_401_CO', '2_FIC_401_PV', '2_FIC_401_SP', '2_FIC_501_CO', '2_FIC_501_PV', '2_FIC_501_SP', '2_FIC_601_CO', '2_FIC_601_PV', '2_FIC_601_SP',
		'2_FQ_101_PV', '2_FQ_201_PV', '2_FQ_301_PV', '2_FQ_401_PV', '2_FQ_501_PV', '2_FQ_601_PV', 
		'2_LS_101_AH', '2_LS_101_AL', '2_LS_201_AH', '2_LS_201_AL', '2_LS_301_AH', '2_LS_301_AL', 
		'2_LS_401_AH', '2_LS_401_AL', '2_LS_501_AH', '2_LS_501_AL', '2_LS_601_AH', '2_LS_601_AL',
		'2_MCV_101_CO', '2_MCV_201_CO', '2_MCV_301_CO', '2_MCV_401_CO', '2_MCV_501_CO', '2_MCV_601_CO',
		'2_MV_101_STATUS', '2_MV_201_STATUS', '2_MV_301_STATUS', '2_MV_401_STATUS', '2_MV_501_STATUS', '2_MV_601_STATUS',
		'2_SV_101_STATUS', '2_SV_201_STATUS', '2_SV_301_STATUS', '2_SV_401_STATUS', '2_SV_501_STATUS', '2_SV_601_STATUS'],
	'Return': ['3_AIT_001_PV', '3_AIT_002_PV', '3_AIT_003_PV', '3_AIT_004_PV', '3_AIT_005_PV', 
		'3_FIT_001_PV', '3_LS_001_AL', '3_LT_001_PV', '3_MV_001_STATUS', '3_MV_002_STATUS', '3_MV_003_STATUS', '3_P_001_STATUS', '3_P_002_STATUS', '3_P_003_STATUS', '3_P_004_STATUS']
   }

   return

def idx_to_sen(idx, sensor_cols):
   return sensor_cols[idx]

def sen_to_idx(sensor, sensor_cols):
   return sensor_cols.index(sensor)

def build_full_wadi_graph():

   dataset_name = 'WADI'

   Xtrain, sensor_cols = load_train_data(dataset_name)
   cov = np.cov(Xtrain[::10].T)

   ###############################
   ### Build Directed Graph
   ###############################
   G = nx.DiGraph()
   G.add_nodes_from(sensor_cols)

   for sidx in range(len(sensor_cols)):
      for didx in range(len(sensor_cols)):
         
         if sidx == didx:
            continue

         src = sensor_cols[sidx]
         dst = sensor_cols[didx]

         if dataset_name == 'WADI-PHY':
            if 'AIT' in src or 'AIT' in dst:
               print(f'Skipping AIT: {src} -> {dst}')
               continue

         G.add_edge(src, dst, weight=np.abs(cov[sidx][didx]))

   nx.write_gml(G, f"explanations-dir/graph-{dataset_name}-full.gml")

def build_proc_wadi_graph():

   dataset_name = 'WADI'

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
      
      if dataset_name == 'WADI-PHY':
         if 'AIT' in src or 'AIT' in dst:
            print(f'Skipping AIT: {src} -> {dst}')
            continue

      sidx = sen_to_idx(src, sensor_cols)
      didx = sen_to_idx(dst, sensor_cols)

      if np.abs(cov[sidx][didx]) == 0:
         print(f'Note: a 0-covariance edge is added {src} {dst}')

      G.add_edge(src, dst, weight=np.abs(cov[sidx][didx]))

   UG = G.to_undirected()
   for i in range(len(sensor_cols)):
      
      sen = sensor_cols[i]
      print(f'graph_nodes: {sen} -> {UG[sen]}')
      print(f'Top 10 related: {[sensor_cols[np.argsort(np.abs(cov[i]))[::-1][j]] for j in range(10)]}')
      print(f'Top 10 covs: {[cov[i][np.argsort(np.abs(cov[i]))[::-1][j]] for j in range(10)]}')

   nx.write_gml(G, f"explanations-dir/graph-{dataset_name}-proc.gml")
   nx.write_gml(UG, f"explanations-dir/graph-ud-{dataset_name}-proc.gml")

def build_third_wadi_graph():

   dataset_name = 'WADI'

   Xtrain, sensor_cols = load_train_data(dataset_name)
   cov = np.cov(Xtrain[::10].T)

   ###############################
   ### Build Directed Graph
   ###############################
   G = nx.DiGraph()
   G.add_nodes_from(sensor_cols)

   subsets = attack_utils.WADI_SUB_MAP

   for i in range(len(PROC_EDGES)):

      src = PROC_EDGES[i][0]
      dst = PROC_EDGES[i][1]

      sidx = sen_to_idx(src, sensor_cols)
      didx = sen_to_idx(dst, sensor_cols)

      if np.abs(cov[sidx][didx]) == 0:
         print(f'Note: a 0-covariance edge is added {src} {dst}')

      G.add_edge(src, dst, weight=np.abs(cov[sidx][didx]))
      G.add_edge(src, dst, weight=np.abs(cov[didx][sidx]))

   for _, sub_name in enumerate(subsets):
      sub_items = subsets[sub_name]
      for si in sub_items:
         for sj in sub_items:
            sidx = sen_to_idx(si, sensor_cols)
            didx = sen_to_idx(sj, sensor_cols)
            G.add_edge(si, sj, weight=np.abs(cov[sidx][didx]))

   pdb.set_trace()

   nx.write_gml(G, f"explanations-dir/graph-{dataset_name}-3rd.gml")

if __name__ == '__main__':

   #build_third_wadi_graph()
   build_proc_wadi_graph()
   #build_full_wadi_graph()

import pdb
import pandas as pd
import numpy as np
import networkx as nx

from . import tep_utils

# Ugly hack for now
import sys
sys.path.append('..')

from data_loader import load_train_data

## From the MATLAB PID
PID_EDGES = [
	('s1', 'a3'),
	('s2', 'a1'),
	('s3', 'a2'),
	('s4', 'a4'),
	('s7', 'a6'),
	('s8', 'a11'),
	('s9', 'a10'),
	('s10', 'a6'),
	('s11', 'a11'),
	('s12', 'a7'),
	('s14', 'a7'),
	('s14', 'a8'),
	('s15', 'a8'),
	('s17', 'a8'),
	('s17', 'a3'),
	('s17', 'a1'),
	('s17', 'a2'),
	('s17', 'a4'),
	('s17', 'a6'),
	('s17', 'a7'),
	('s23', 'a3'),
	('s23', 'a4'),
	('s25', 'a3'),
	('s25', 'a4'),
	('s40', 'a1'),
	('s40', 'a2'),

	# Actuator to sensor
	('a3', 's1'),
	('a1', 's2'),
	('a2', 's3'),
	('a4', 's4'),
	#('a5', 's5'),
	#('a5', 's20'),
	('a6', 's10'),
	('a7', 's14'),
	('a8', 's7'),
	#('a9', 's19'),
	('a10', 's21'),
	('a11', 's22')

]

PROC_EDGES = [

	# Input flows
	('s1', 's6'),
	('s2', 's6'),
	('s3', 's6'),

	# Reactor process
	('s6', 's7'),
	('s6', 's8'),
	('s6', 's9'),
	('s6', 's21'),

	# Condenser + Separator
	('s7', 's22'),
	('s7', 's11'),
	('s7', 's12'),
	('s7', 's13'),
	('s11', 's14'),
	('s12', 's14'),

	# Stripper
	('s4', 's15'),
	('s4', 's18'),
	('s14', 's15'),
	('s14', 's18'),
	('s19', 's15'),
	('s19', 's18'),
	('s15', 's16'),
	('s18', 's16'),
	('s15', 's17'),
	('s18', 's17'),
	('s16', 's5'),

	# Compressor
	('s13', 's5'),
	('s13', 's20'),
	('s13', 's10'),

	# Reactor Flow
	('s5', 's6'),
	('s6', 's23'),
	('s6', 's24'),
	('s6', 's25'),
	('s6', 's26'),
	('s6', 's27'),
	('s6', 's28'),

	# Purge
	('s10', 's29'),
	('s10', 's30'),
	('s10', 's31'),
	('s10', 's32'),
	('s10', 's33'),
	('s10', 's34'),
	('s10', 's35'),
	('s10', 's36'),

	# Product
	('s17', 's37'),
	('s17', 's38'),
	('s17', 's39'),
	('s17', 's40'),
	('s17', 's41')

]

def get_sensor_subsets_name():

	sensor_groups_by_name = [
		[5, 6, 7, 8, 20, 22, 23, 24, 25, 26, 27, 50],
		[9, 28, 29, 30, 31, 32, 33, 34, 35, 46],
		[10, 11, 12, 13, 36, 37, 38, 39, 40],
		[14, 15, 16, 17, 18, 48],
		[21, 47],
		[0, 1, 2, 3, 41, 42, 43, 44],
		[4, 45],
		[19],
		[49], 
		[51], 
		[52],
	]

	labels_by_name = [
		['Reactor Feed Rate', 'Reactor Pressure', 'Reactor Level', 'Reactor Temperature', 'Reactor Coolant Temp', 'Comp A to Reactor', 'Comp B to Reactor', 'Comp C to Reactor', 'Comp D to Reactor', 'Comp E to Reactor', 'Comp F to Reactor', 'Reactor Coolant'],
		['Purge Rate', 'Comp A in Purge', 'Comp B in Purge', 'Comp C in Purge', 'Comp D in Purge', 'Comp E in Purge', 'Comp F in Purge', 'Comp G in Purge', 'Comp H in Purge', 'Purge'],
		['Product Sep Temp', 'Product Sep Level', 'Product Sep Pressure', 'Product Sep Underflow', 'Comp D in Product', 'Comp E in Product', 'Comp F in Product', 'Comp G in Product', 'Comp H in Product'],
		['Stripper Level', 'Stripper Pressure', 'Stripper Underflow', 'Stripper Temp', 'Stripper Steam Flow', 'Stripper'],
		['Separator Coolant Temp', 'Separator'],
		['A Feed', 'D Feed', 'E Feed', 'A and C Feed', 'D feed', 'E Feed.1', 'A Feed.1', 'A and C Feed.1'],
		['Recycle Flow', 'Recycle'], 
		['Compressor Work'],
		['Steam'], 
		['Condenser Coolant (MV)'], 
		['Agitator'],
	]

	return sensor_groups_by_name, labels_by_name

# Sensor subsets using simulink connections
def get_sensor_subsets_simulink():

	sensor_groups_by_plc = [
		[1, 16, 39, 41],     # XMV 1
		[2, 16, 39, 42],     # XMV 2
		[0, 16, 22, 24, 43], # XMV 3
		[3, 16, 22, 24, 44], # XMV 4
		[6, 9, 16, 46],      # XMV 6
		[11, 13, 16, 47],    # XMV 7
		[14, 16, 48],        # XMV 8
		[8, 50],             # XMV 10
		[7, 10, 51],         # XMV 11
	]

	labels_by_plc = [
		['D Feed', 'Stripper Underflow', 'Comp G in Product', 'D Feed (MV)'],
		['E Feed', 'Stripper Underflow', 'Comp G in Product', 'E Feed (MV)'],
		['A Feed', 'Stripper Underflow', 'Comp A to Reactor', 'Comp C to Reactor', 'A Feed (MV)'],
		['A and C Feed', 'Stripper Underflow', 'Comp A to Reactor', 'Comp C to Reactor', 'A and C Feed (MV)'],
		['Reactor Pressure', 'Purge Rate', 'Stripper Underflow', 'Purge (MV)'],
		['Product Sep Level', 'Product Sep Underflow', 'Stripper Underflow', 'Separator (MV)'],
		['Stripper Level', 'Stripper Underflow', 'Stripper (MV)'],
		['Reactor Temperature', 'Reactor Coolant (MV)'],
		['Reactor Level', 'Product Sep Temp', 'Condenser Coolant (MV)'],
	]

	return sensor_groups_by_plc, labels_by_plc

def build_proc_graph():

	dataset_name = 'TEP'

	Xtrain, _ = load_train_data(dataset_name)
	cov = np.cov(Xtrain[::4].T)

	sensor_cols = tep_utils.get_short_colnames()

	###############################
	### Build Directed Graph
	###############################
	G = nx.DiGraph()
	G.add_nodes_from(sensor_cols)

	ALL_EDGES = PID_EDGES + PROC_EDGES

	for i in range(len(ALL_EDGES)):

		src = ALL_EDGES[i][0]
		dst = ALL_EDGES[i][1]
	  
		sidx = tep_utils.sen_to_idx(src)
		didx = tep_utils.sen_to_idx(dst)

		if np.abs(cov[sidx][didx]) == 0:
			print(f'Note: a 0-covariance edge is being added: {src} {dst}')

		#G.add_edge(src, src, weight=np.abs(cov[sidx][sidx]))
		#G.add_edge(dst, src, weight=np.abs(cov[sidx][didx]))
		G.add_edge(src, dst, weight=np.abs(cov[sidx][didx]))

	UG = G.to_undirected()

	#pdb.set_trace()

	nx.write_gml(G, f"explanations-dir/graph-{dataset_name}-all.gml")
	nx.write_gml(UG, f"explanations-dir/graph-ud-{dataset_name}-all.gml")

if __name__ == '__main__':

	build_proc_graph()

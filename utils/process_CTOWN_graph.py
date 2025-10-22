import pdb
import pandas as pd
import numpy as np
import networkx as nx

# Ugly hack for now
import sys
sys.path.append('..')

ALL_SENSORS = ['PU1F', 'PU2F', 'J280', 'J269', 'PU1', 'PU2', 'T1', 'T2', 'V2F', 'J300', 'J256', 'J289', 'J415', 'J14', 'J422', 
 	'PU4F', 'PU5F', 'PU6F', 'PU7F', 'V2', 'PU4', 'PU5', 'PU6', 'PU7', 'T3', 'PU8F', 'PU10F', 'PU11F', 'J302', 
	'J306', 'J307', 'J317', 'PU8', 'PU10', 'PU11', 'T4', 'T5', 'T6', 'T7']

PROC_EDGES = [

	# Tank Control
	('T1', 'PU1'),
	('T1', 'PU2'),
	('T2', 'V2'),
	('T3', 'PU4'),
	('T3', 'PU5'),
	('T4', 'PU6'),
	('T4', 'PU7'),
	('T5', 'PU8'),
	('T7', 'PU10'),
	('T7', 'PU11'),

	# Pump Flows
	('PU1', 'PU1F'),
	('PU2', 'PU2F'),
	('PU4', 'PU4F'),
	('PU5', 'PU5F'),
	('PU6', 'PU6F'),
	('PU7', 'PU7F'),
	('PU8', 'PU8F'),
	('PU10', 'PU10F'),
	('PU11', 'PU11F'),
	
	# Pipes with pumps
	('PU2F', 'J280'),
	('PU2F', 'J269'),
	('J280', 'J269'),

	('PU5F', 'J300'),
	('PU5F', 'J256'),
	('J300', 'J256'),
	
	('PU6F', 'J289'),
	('PU6F', 'J415'),
	('J289', 'J415'),
	('J302', 'J306'), # Pump 9 not in SCADA
	
	('PU10F', 'J307'),
	('PU10F', 'J317'),
	('J307', 'J317'),

	# Valve 2
	('V2', 'J14'),
	('V2', 'J422'),
	('J14', 'J422')
]

def get_sensor_subsets():

   subprocess_idxs = [
      [0, 1, 2, 3, 4, 5], 		# PLC 1
      [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], # PLC 3
      [25, 26, 27, 28, 29, 30, 31, 32, 33, 34], # PLC 5
   ]

   subprocess_labels = [
    	['PU1F', 'PU2F', 'J280', 'J269', 'PU1', 'PU2'],
    	['T2', 'V2F', 'J300', 'J256', 'J289', 'J415', 'J14', 'J422', 'PU4F', 'PU5F', 'PU6F', 'PU7F', 'V2', 'PU4', 'PU5', 'PU6', 'PU7'], 
    	['PU8F', 'PU10F', 'PU11F', 'J302', 'J306', 'J307', 'J317', 'PU8', 'PU10', 'PU11'],
   ]

   return subprocess_idxs, subprocess_labels

def idx_to_sen(idx, sensor_cols):
   return sensor_cols[idx]

def sen_to_idx(sensor, sensor_cols):
   return sensor_cols.index(sensor)

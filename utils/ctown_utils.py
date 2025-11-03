import os
import numpy as np
import pandas as pd
import pickle

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

import pdb

# NOTE: Removed 6 and 8, since no naive versions of them exist
# NOTE: Removed 14 and 40, due to NaN values
NAIVE_ATTACKS = [38,45,39,41,42,43,46]

# NOTE: added all concealment versions for the naive attacks
ORIGINAL_CONCEAL_ATTACKS = [1,3,11,25,27,34,36]
CONCEAL_ATTACKS = [1,2,3,4,5,11,12,13,25,26,27,28,29,34,36,37,44]

# VALUE_ATTACKS = [1,3,6,8,11,25,27,34,36]
# SIMPLE_ATTACKS = [1,3,11,25,34,36]

def idx_to_sen(idx, sensor_cols):
   return sensor_cols[idx]

def sen_to_idx(sensor, sensor_cols):
   return sensor_cols.index(sensor)

def get_attack_start_and_end(attack_number):
	
	attack_start = -1
	attack_end = -1
	
	# Referencing table 3 in RICSS paper
	if attack_number in [38, 45, 39, 1, 2, 44, 3, 4, 5, 11, 12, 13]:
		attack_start = 1440
		attack_end = 1812
	elif attack_number in [41, 42, 25, 26, 27, 28, 29]:
		attack_start = 1600
		attack_end = 2728
	elif attack_number in [43, 46, 34, 35, 36, 37]:
		attack_start = 1368
		attack_end = 1728

	return attack_start, attack_end

# NOTE: Only including the real attacks ("device attack" or "network attack") as correct answers
# Other manipulations are performed as concealment, but aren't part of the physical harm
def get_attack_column_labels(stealth=None):
	
	if stealth == 'conceal':

		# NOTE: added all concealment versions for the naive attacks
		alt_labels = [
			['PU10', 'PU11'], 											# Attack 38 > 1, attack PU10/PU11, conceal with T7 
			['PU10', 'PU11'], 											# Attack 38 > 2, attack PU10/PU11, conceal with T7 
			['T7'], 													# Attack 45 > 3, just concealment_mitm on T7
			['T7'], 													# Attack 45 > 4, just concealment_mitm on T7
			['T7'], 													# Attack 45 > 5, just concealment_mitm on T7
			# ['PU10', 'PU11'], 										# Attack 6, attack PU10/PU11, conceal on all 5
			# ['T7'], 													# Attack 8, attack T7, conceal on all 5
			['T1'],														# Attack 39 > 11, just concealment_mitm on T1
			['T1'],														# Attack 39 > 12, just concealment_mitm on T1
			['T1'],														# Attack 39 > 13, just concealment_mitm on T1
			#['T1'],  													# Attack 40 > 14, attack T1, conceal on all 7
			['V2'],  													# Attack 41 > 25, attack V2, conceal with T2
			['V2'],  													# Attack 41 > 26, attack V2, conceal with T2
			['V2'], 							 						# Attack 42 > 27, attack V2, conceal with other 4
			['V2'], 							 						# Attack 42 > 28, attack V2, conceal with other 4
			['V2'], 							 						# Attack 42 > 29, attack V2, conceal with other 4
			['T4'],  													# Attack 43 > 34, just concealment_mitm on T4
			#['T4'],  													# Attack 43 > 35, just concealment_mitm on T4 (longer than 2880)
			['PU6', 'PU7'],  											# Attack 46 > 36, attack PU6/PU7, conceal with T4
			['PU6', 'PU7'],  											# Attack 46 > 37, attack PU6/PU7, conceal with T4
			['PU10', 'PU11'], 											# Attack 38 > 44, attack PU10/PU11, conceal with T7 
		]

		return CONCEAL_ATTACKS, alt_labels
	
	elif stealth == 'replay_target':

		# Eval against the full set of concealment
		replay_target_labels = [
			['PU10', 'PU11'], 										# Attack 38, attack PU10/PU11
			['T7'], 												# Attack 45, MitM on T7
			['T1'],													# Attack 39, MitM on T1
			#['T1'],												# Attack 40, MitM on T1
			['V2'],  												# Attack 41, attack V2
			['V2'],  												# Attack 42, attack V2
			['T4'],  												# Attack 43, MitM on T4
			['PU6', 'PU7']  										# Attack 46, attack PU6/PU7
		]
	
		return NAIVE_ATTACKS, replay_target_labels

	elif stealth == 'replay_plc':

		# Eval against custom PLC spoofing
		replay_plc_labels = [
			['PU10', 'PU11'], 										# Attack 38, attack PU10/PU11
			['T1'],													# Attack 39, MitM on T1
			#['T1'],												# Attack 40, MitM on T1
			['V2'],  												# Attack 41, attack V2
			['V2'],  												# Attack 42, attack V2
			['PU6', 'PU7']  										# Attack 46, attack PU6/PU7
		]
	
		replay_plc_numbers = [38,39,41,42,46]

		return replay_plc_numbers, replay_plc_labels

	else:
		
		# Eval against custom target spoofing
		labels = [
			['PU10', 'PU11'], 										# Attack 38, attack PU10/PU11
			['T7'], 												# Attack 45, MitM on T7
			['T1'],													# Attack 39, MitM on T1
			#['T1'],													# Attack 40, MitM on T1
			['V2'],  												# Attack 41, attack V2
			['V2'],  												# Attack 42, attack V2
			['T4'],  												# Attack 43, MitM on T4
			['PU6', 'PU7']  										# Attack 46, attack PU6/PU7
		]

	return NAIVE_ATTACKS, labels

def load_ctown_attack(dataset_name, attack_number, spoofing=None, scaler=None, no_transform=False, verbose=1):

	if verbose > 0:
		print(f'Loading {dataset_name} attack {attack_number}...')

	if scaler is None:
		if verbose > 0:
			print('No scaler provided, loading from models directory.')
		scaler = pickle.load(open(f'checkpoints/CTOWN_multi_scaler.pkl', "rb"))

	if dataset_name == 'CTOWN':

		if spoofing == None:
			# Directly Load data from prior work repository
			df_test = pd.read_csv(f"/pwwl/Practical-Evasion-Attacks/dataset/evasion_data/attack_output_{attack_number:02d}/scada_values.csv")
			sensor_cols = [col for col in df_test.columns if col not in ['iteration', 'timestamp', 'Attack']]

		else:
			# Load our version of the attack, after using the spoofing script (main_spoofing_ctown.py)
			df_test = pd.read_csv(f"data/CTOWN/CTOWN_attack{attack_number:02d}_{spoofing}.csv")
			sensor_cols = [col for col in df_test.columns if col not in ['iteration', 'timestamp', 'Attack']]

	else:
		print('This function (from ctown_utils.py) is meant for CTOWN only.')
		return

	# scale sensor data
	if no_transform:
		Xtest = pd.DataFrame(index = df_test.index, columns = sensor_cols, data = df_test[sensor_cols])
	else:
		Xtest = pd.DataFrame(index = df_test.index, columns = sensor_cols, data = scaler.transform(df_test[sensor_cols]))
	
	assert(len(Xtest) == 2881)

	# Manually construct Ytest
	start, end = get_attack_start_and_end(attack_number)
	Ytest = np.zeros(2881)
	Ytest[start:end] = 1

	return Xtest.values, Ytest, sensor_cols

class CTownMultiWeekDataset(Dataset):
	
	def __init__(self, config):
		
		self.num_files = 52
		self.csv_root = "/pwwl/Practical-Evasion-Attacks/dataset/normal_operating_conditions_3_0/output"
		self.time_window = config['history']
		self.scaler = StandardScaler()

		# Build an index mapping: dataset index -> (csv_file_index, row_in_file)
		self.all_dfs = []
		self.headers = None

		for file_idx in range(self.num_files):

			df = pd.read_csv(f"{self.csv_root}/batch_{file_idx}/scada_values.csv")
			sensor_cols = [col for col in df.columns if col not in ['iteration', 'timestamp']]
			
			# We omit the first segment of history from each file
			self.all_dfs.append(df[sensor_cols].values)

		self.rows_per_file = len(df) - self.time_window
		self.Xprescaled = np.stack(self.all_dfs, axis=0)
		self.scaler.fit(np.concatenate(self.Xprescaled, axis=0))
		pickle.dump(self.scaler, open(f'checkpoints/CTOWN_multi_scaler.pkl', 'wb'))

	def __len__(self):
		return self.rows_per_file * self.num_files

	def __getitem__(self, idx):
		
		file_idx = idx // self.rows_per_file
		local_idx = (idx % self.rows_per_file) + self.time_window

		X_inner = self.Xprescaled[file_idx]

		history = self.scaler.transform(X_inner[local_idx - self.time_window : local_idx])
		target = self.scaler.transform(X_inner[local_idx].reshape(1, -1))
		label = 0

		# Some ugly transforms for complicance
		features_out = torch.tensor(history.T).double()
		target_out = torch.tensor(target[0]).double()
		labels_out = torch.tensor(label).double()

		return features_out, target_out, labels_out

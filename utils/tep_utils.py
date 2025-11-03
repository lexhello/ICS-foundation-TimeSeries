import os
import numpy as np
import pandas as pd
import pickle

import pdb

def get_short_colnames():

	col_names = []
	for i in range(53):
		col_names.append(idx_to_sen(i))

	return col_names

def get_xmv():

	xmv_attack_numbers = ['a1', 'a2', 'a3', 'a4', 'a6', 'a7', 'a8', 'a10', 'a11'] #9 total
	return xmv_attack_numbers

def get_non_pid():

	non_pid_sensors = ['s5', 's6', 's13', 's16', 's18', 's19', 's20']
	return non_pid_sensors

def get_pid():
	
	pid_sensors = ['s1', 's2', 's3', 's4', 's7', 's8', 's9', 's10', 's11', 's12', 's14', 's15', 's17', 's23', 's25', 's40'] #16 total
	return pid_sensors

# Get PID features that are within one PLC only
def get_solo_pid():
	
	pid_sensors = ['s1', 's2', 's3', 's4', 's7', 's8', 's9', 's10', 's11', 's12', 's14', 's15'] #12 total
	return pid_sensors

def get_skip_list():
	
	skip_list = [
		'cons_p2s_s4', 
		'cons_p2s_s9',
		'cons_p2s_a11',

		####
		# These do not crash the system, but we should skip them for fair comparison
		'csum_p2s_s4', 
		'csum_p2s_s9',
		'csum_p2s_a11',
		'line_p2s_s4', 
		'line_p2s_s9',
		'line_p2s_a11',
		'lsum_p2s_s4', 
		'lsum_p2s_s9',
		'lsum_p2s_a11',
		####

		'cons_p3s_s4', 
		'cons_p3s_s9',
		'cons_p3s_a11', 
		'cons_p5s_s4', 
		'cons_p5s_s9',
		'cons_p5s_a11', 
		'cons_p5s_s3', 
		'cons_p5s_s17',
		'line_p3s_s9',
		'line_p5s_s9',
		'line_p5s_a11',
		]
	
	return skip_list

def get_footer_list(patterns=None, mags=None, locations=None):

	if locations == None:
		feature_list = get_pid() + get_xmv()
	elif locations == 'pid':
		feature_list = get_pid()
	elif locations == 'nonpid':
		feature_list = get_non_pid()
	elif locations == 'xmv':
		feature_list = get_xmv()
	elif locations == 'replay_target':
		feature_list = get_pid() + get_xmv()
	elif locations == 'replay_plc':
		feature_list = get_solo_pid() + get_xmv()

	if patterns is None:
		attack_patterns = ['cons']
	else:
		attack_patterns = patterns

	if mags is None:
		attack_mags = ['p2s', 'm2s', 'p3s', 'p5s']
	else:
		attack_mags = mags

	footers = []
	for am in attack_mags:
		for ap in attack_patterns:
			for loc in feature_list:
					footer = f'{ap}_{am}_{loc}'
					if footer not in get_skip_list():
						fname = f"/pwwl/tep-attacks/matlab/TEP_test_{footer}.csv"
						if os.path.isfile(fname):
							footers.append(footer)
						else:
							print(f'could not find {footer}')

	return footers

def sen_to_idx(sensor):

	sensor_type = sensor[0]
	sensor_value = int(sensor[1:])

	if sensor_type == 'a':
		return sensor_value + 40
	elif sensor_type == 's':
		return sensor_value - 1

def idx_to_sen(idx):

	if idx > 40:
		return f'a{idx-40}'
	return f's{idx+1}'

def check_subsystems(sensor_cols):

	reactor_list = []
	purge_list = []
	product_list = []
	stripper_list = []
	separator_list = []
	feed_list = []
	other_list = []

	for col_name in sensor_cols:

		idx = sensor_cols.index(col_name)
		print(f'{col_name} is {idx_to_sen(idx)} at {idx}')

		if "Reactor" in col_name:
			reactor_list.append(idx)
		elif "Purge" in col_name:
			purge_list.append(idx)
		elif "Product" in col_name:
			product_list.append(idx)
		elif "Stripper" in col_name:
			stripper_list.append(idx)
		elif "Separator" in col_name:
			separator_list.append(idx)
		elif "eed" in col_name:
			feed_list.append(idx)
		else:
			other_list.append(idx)

	print(reactor_list)
	print(purge_list)
	print(product_list)
	print(stripper_list)
	print(separator_list)
	print(feed_list)
	print(other_list)

# Get a set of relevance_scores and predicted scores
def calc_dcg(pred_scores, rel_scores, p=10):

	# Take the indices of the highest p predicted scores
	rankings = np.argsort(pred_scores)[::-1]
	total_score = 0

	for i in range(p):
		rank_i_idx = rankings[i]
		score = (np.power(2, rel_scores[rank_i_idx]) - 1) / np.log2(i + 2)
		total_score += score

	return total_score

# Same as DCG, but include normalization to [0-1] based on ideal score
def calc_ndcg(pred_scores, rel_scores, p=10):
	
	score = calc_dcg(pred_scores, rel_scores, p)
	ideal_score = calc_dcg(rel_scores, rel_scores, p)
	return score / ideal_score

def attack_footer_to_sensor_idx(attack_footer):

	splits = attack_footer.split("_")
	sensor_type = splits[2][0]
	sensor_value = int(splits[2][1:])

	if sensor_type == 'a':
		return sensor_value + 40
	elif sensor_type == 's':
		return sensor_value - 1
	else:
		print(f'Something wrong! Found sensor_type {sensor_type}')
		exit()

	return -1

def load_tep_attack(dataset_name, attack_footer, spoofing=None, scaler=None, no_transform=False, verbose=1):

	if verbose > 0:
		print(f'Loading {dataset_name} attack {attack_footer}...')

	if scaler is None:
		if verbose > 0:
			print('No scaler provided, loading from models directory.')
		scaler = pickle.load(open(f'checkpoints/{dataset_name}_scaler.pkl', "rb"))

	if dataset_name == 'TEP':

		if spoofing == None:
			# Directly Load data from prior work repository
			df_test = pd.read_csv(f"/pwwl/tep-attacks/matlab/TEP_test_{attack_footer}.csv", dayfirst=True)
		else:
			# Load our version of the attack, after using the spoofing script (main_spoofing_tep.py)
			df_test = pd.read_csv(f"data/TEP/TEP_attack_{attack_footer}_{spoofing}.csv", dayfirst=True)
		
		sensor_cols = [col for col in df_test.columns if col not in ['Atk']]
		target_col = 'Atk'

	else:
		print('This script is meant for TEP only.')
		return

	# scale sensor data
	if no_transform:
		Xtest = pd.DataFrame(index = df_test.index, columns = sensor_cols, data = df_test[sensor_cols])
	else:
		Xtest = pd.DataFrame(index = df_test.index, columns = sensor_cols, data = scaler.transform(df_test[sensor_cols]))
	
	Ytest = df_test[target_col]
	
	return Xtest.values[:14000], Ytest.values[:14000], get_short_colnames()

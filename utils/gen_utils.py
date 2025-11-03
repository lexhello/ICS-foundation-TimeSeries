import argparse
import numpy as np
from sklearn.model_selection import train_test_split

def normalize_array_length(arr1, arr2):

	if len(arr1) < len(arr2):
		cut_amt = len(arr2) - len(arr1)
		return arr1, arr2[cut_amt:]
	
	elif len(arr1) > len(arr2):
		cut_amt = len(arr1) - len(arr2)
		return arr1[cut_amt:], arr2
	
	else:
		return arr1, arr2

	return arr1, arr2        

def build_baseline(Xfull, history, method='MSE'):

	# Build via indexs
	all_idxs = np.arange(history, len(Xfull)-1)
	train_idxs, val_idxs, _, _ = train_test_split(all_idxs, all_idxs, test_size=0.2, random_state=42, shuffle=True)

	Xbatch = []

	# Build the history out by sampling from the list of idxs
	for b in range(5000):
		lead_idx = val_idxs[b]
		Xbatch.append(Xfull[lead_idx-history:lead_idx])

	Xbatch = np.array(Xbatch)

	if method == 'IG' or method == 'CF-Add' or method == 'CF-Sub':

		#################################
		# BASELINE FOR INTEGRATED GRADIENTS
		#################################

		avg_benign = np.mean(Xbatch, axis=0)
		baseline = np.expand_dims(avg_benign, axis=0)
		print(f'Built {method} baseline obj')

	elif method == 'EG' or method == 'LIME' or method == 'SHAP':

		#################################
		# BASELINE FOR EXPECTED GRAD
		#################################

		# If using expected gradients, baseline needs to be training examples. We subsample training regions
		train_samples = []
		num_to_sample = 500
		inc = len(Xbatch) // num_to_sample

		for i in range(num_to_sample):
			train_sample = Xbatch[i * inc]
			train_samples.append(train_sample)

		baseline = np.array(train_samples)
		print(f'Built {method} baseline obj')

	else:

		baseline = None

	return baseline

def train_val_history_idx_split(dataset_name, Xfull, history, train_size=0.8, shuffle=True):
	
	val_size = 1 - train_size
	all_idxs = np.arange(history, len(Xfull)-1)
	train_idxs, val_idxs, _, _ = train_test_split(all_idxs, all_idxs, test_size=val_size, random_state=42, shuffle=shuffle)	
	return train_idxs, val_idxs

def custom_train_test_split(dataset_name, Xtest, Ytest, test_size=0.7, shuffle=False):

	if dataset_name == 'BATADAL':
		# The first 30% of BATADAL contains no attacks, so we use the back 30% instead
		Xtest_test, Xtest_val, Ytest_test, Ytest_val = train_test_split(Xtest, Ytest, test_size=1-test_size, shuffle=shuffle)
	else:
		Xtest_val, Xtest_test, Ytest_val, Ytest_test = train_test_split(Xtest, Ytest, test_size=test_size, shuffle=shuffle)

	return Xtest_val, Xtest_test, Ytest_val, Ytest_test

def transform_to_window_data(dataset, target, history, target_size=1):
		data = []
		targets = []

		start_index = history
		end_index = len(dataset) - target_size

		for i in range(start_index, end_index):
			indices = range(i - history, i)
			data.append(dataset[indices])
			targets.append(target[i+target_size])

		return np.array(data), np.array(targets)

# Generic data generator object for feeding data
def reconstruction_errors_by_idxs(event_detector, Xfull, idxs, history, bs=4096):
    
    # Length of reconstruction errors is len(X) - history. Clipped from the front.
    full_errors = np.zeros((len(idxs), Xfull.shape[1]))
    idx = 0

    for idx in range(0, len(idxs), bs):
        
        Xbatch = []
        Ybatch = []

        # Build the history out by sampling from the list of idxs
        for b in range(bs):
            
            if idx + b >= len(idxs):
                break
            
            lead_idx = idxs[idx+b]
            Xbatch.append(Xfull[lead_idx-history:lead_idx])
            Ybatch.append(Xfull[lead_idx+1])

        Xbatch = np.array(Xbatch)
        Ybatch = np.array(Ybatch)

        if idx + bs > len(full_errors):
            full_errors[idx:] = (event_detector.predict(Xbatch) - Ybatch)**2                
        else:
            full_errors[idx:idx+bs] = (event_detector.predict(Xbatch) - Ybatch)**2

    return full_errors

def get_argparser():
	
	parser = argparse.ArgumentParser()
	
	parser.add_argument("model", 
		help="Type of model to use",
		choices=['GDN', 'FSN', 'CNN', 'GRU', 'LSTM', 'CPDD', 'AR', 'GCN', 'TGDDPLC', 'TGDDPROC'],
		default='GDN')

	parser.add_argument("dataset", 
		help="Dataset to use",
		choices=['SWAT', 'WADI', 'TEP', 'CTOWN'],
		default='TEP')

	parser.add_argument("--gpus", 
		help="GPUs to use",
		type=str,
		default="-1")

	parser.add_argument("--model_params_history", 
		default=10,
		type=int,
		help="History/window parameter from AAAI'21 implementation. Describe how many timesteps are used for prediction.")

	### Model specific params
	parser.add_argument("--gnn_model_params_topk", 
		default=20,
		type=int,
		help="TopK parameter from AAAI'21 implementation. Used to filter attention weights into graph edges, higher K results in denser graphs.")

	parser.add_argument("--gcn_model_params_degree", 
		default=1,
		type=int,
		help="Degree in GCN model (i.e., Do a 2nd-order passes)")
	
	parser.add_argument('--gcn_model_params_weight_strategy', 
        default='keep',
        type=str,
        help='Keep weights in GCN. Default keeps all weights')

	parser.add_argument('--gcn_graph_spec', 
		type=str,
        help="Specification for a graph. Can be a model name, 'plc', 'subprocess', or 'full'.")

	### NNs
	parser.add_argument("--nn_model_params_units", 
		default=64,
		type=int,
		help="Number of units per layer in NN model")
	
	parser.add_argument("--nn_model_params_layers", 
		default=4,
		type=int,
		help="Number layers in NN model")

	return parser

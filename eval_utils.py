
import numpy as np
import pdb
import torch
import sys

from tqdm import tqdm
from utils import tep_utils, attack_utils, ctown_utils, metrics
import utils.attack_utils
from torch.utils.data import DataLoader
from data_loader import TimeSeriesDataset, load_test_data

from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, roc_curve
#TODO: delete
import importlib

importlib.reload(utils.attack_utils)


LIST_OF_THRESHOLDS = ['best-auroc-threshold', 'fpr1-threshold', 'fpr5-threshold', 
                    'mse-val995-threshold', 'mse-val1-threshold', 'cusum-total-threshold', 'cusum-pf-threshold',
                    'mse-pf-val995-threshold', 'mse-pf-val1-threshold']

def run_slim_test(device, model, dataloader):

    # test
    loss_func = torch.nn.MSELoss(reduction='mean')
    test_loss_list = []

    model.eval()

    # Peek at the first item in the dataset to get the n_sensors
    n_sensors = dataloader.dataset.__getitem__(0)[0].shape[0]
    t_test_predicted_list = np.zeros((len(dataloader), n_sensors))
    t_test_ground_list = np.zeros((len(dataloader), n_sensors))
    t_test_labels_list = np.zeros(len(dataloader))

    i = 0
    acu_loss = 0
    for x, y, labels in tqdm(dataloader):
        
        x, y, labels = [item.to(device).float() for item in [x, y, labels]]
        
        with torch.no_grad():
            
            predicted = model(x).float().to(device)
            loss = loss_func(predicted, y)

            t_test_predicted_list[i] = predicted.cpu().numpy()
            t_test_ground_list[i] = y.cpu().numpy()
            t_test_labels_list[i] = labels.cpu().numpy()

        test_loss_list.append(loss.item())
        acu_loss += loss.item()
        i += 1

    avg_loss = sum(test_loss_list)/len(test_loss_list)

    return avg_loss, [t_test_predicted_list, t_test_ground_list, t_test_labels_list]

def run_times_series_test(model_name, model, val_dataloader, cfg):
    
    loss_func = torch.nn.MSELoss(reduction='mean')
    
    test_loss_list = []
    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []
    
    max_samples = 5
    if model_name == "timesfm" or model_name == "TimesFM":
        
        print("USING TIMESFM")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"
        print(f'On device: {device}')
        
        count = 0
        for x, y, labels in tqdm(val_dataloader):
            
            count+=1
            if count > max_samples:
                break
            # print("INSIDE A BATCH")
            x = x.squeeze(0)
            y = y.T
            point_forecast, quantile_forecast = model.forecast(
                horizon=1,
                inputs=x,  # Two dummy inputs
            )
            # print("shape of point_forecast")
            # print(point_forecast.shape)
            # print("shape of y")
            # print(y.shape)

            predictions = torch.tensor(point_forecast, dtype=torch.float32)
             # torch.tensor(y, dtype=torch.float32)
            predictions = predictions.view(1, -1)  # ensures [1, 50]
            y = y.view(1, -1)                      # ensures [1, 50]
            target = y
            
            loss = loss_func(predictions, target)
            
            test_loss_list.append(loss.item())
            
            if len(t_test_predicted_list) == 0:
                t_test_predicted_list = predictions
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predictions), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)


    # t_test_predicted_list = torch.stack(t_test_predicted_list, dim=0)
    # t_test_ground_list = torch.stack(t_test_ground_list, dim=0)
    # t_test_labels_list = torch.stack(t_test_labels_list, dim=0)

    print("eval list shape:")
    # i think this is wrong 
    print(t_test_predicted_list.shape)
    # Convert to NumPy arrays
    test_predicted_list = t_test_predicted_list.cpu().numpy()
    test_ground_list = t_test_ground_list.cpu().numpy()
    test_labels_list = t_test_labels_list.cpu().numpy()
    
    avg_loss = sum(test_loss_list)/len(test_loss_list)
    
    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]
        

def run_test(device, model, dataloader, sensor_cols):
    
    if dataloader.batch_size == 1:
        avg_loss, test_result = run_slim_test(device, model, dataloader)
        return avg_loss, test_result

    # test
    loss_func = torch.nn.MSELoss(reduction='mean')

    test_loss_list = []
    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    model.eval()

    i = 0
    acu_loss = 0
    for x, y, labels in tqdm(dataloader):
        
        x, y, labels = [item.to(device).float() for item in [x, y, labels]]
        
        with torch.no_grad():
            
            predicted = model(x).float().to(device)
            loss = loss_func(predicted, y)

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)
        
        test_loss_list.append(loss.item())
        acu_loss += loss.item()
        i += 1

    test_predicted_list = t_test_predicted_list.cpu().numpy()        
    test_ground_list = t_test_ground_list.cpu().numpy()
    test_labels_list = t_test_labels_list.cpu().numpy()
    
    avg_loss = sum(test_loss_list)/len(test_loss_list)

    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]

# TODO: if needed, add the hyperparameters used by GeCo: scale factor S and growth factor G.
def cumulative_sum(Xpred, Xtrue, drift=0, per_feature=False):

    # Do a per feature cumulative sum, assume drift is 1xd
    if per_feature:
        
        n_sensors = Xtrue.shape[1]
        cusum = np.zeros((len(Xtrue), n_sensors))
        
        for j in range(n_sensors):
            
            cusum[0, j] = 0
            for i in range(1, len(Xtrue)):
                cusum[i, j] = max(0, cusum[i-1, j] + np.abs(Xtrue[i, j] - Xpred[i, j]) - drift[j])

        return cusum

    else:
        print("shape of xtrue and xpred in cusum calc")
        print(Xtrue.shape, Xpred.shape)
        print("shape of drift: ")
        print(drift.shape)
        
        cusum = np.zeros(len(Xtrue))
        cusum[0] = 0

        for i in range(1, len(Xtrue)):
            cusum[i] = max(0, cusum[i-1] + np.sum(np.abs(Xtrue[i] - Xpred[i])) - drift)

    return cusum

def get_attacks_info(dataset, evasion_type):

    # List of tuples: (attack_name, col_name)
    attacks_obj = list()

    if dataset == 'TEP':

        if evasion_type == 'replay_plc' or evasion_type == 'replay_target':
            all_attacks = tep_utils.get_footer_list(patterns=['cons'], mags=['p2s'], locations=evasion_type)    
        else:
            all_attacks = tep_utils.get_footer_list(patterns=[evasion_type], mags=['p2s'], locations=None)
        
        for footer in all_attacks:
            splits = footer.split("_")
            col_name = splits[2]
            attacks_obj.append((footer, col_name))

    elif dataset == 'CTOWN':

        numbers, labels = ctown_utils.get_attack_column_labels(stealth=evasion_type)
        
        for i in range(len(numbers)):
            attacks_obj.append((f'{dataset}_{numbers[i]}', labels[i]))

    elif dataset in ['SWAT', 'WADI']:
        indices, labels = attack_utils.get_attack_indices(dataset)
        #TODO
        print("LEN OF INDICES")
        print(len(indices))
        
        
        for i in range(len(indices)):
            attacks_obj.append((f'{dataset}_{i}', labels[i]))

    print("len of attacks_obj:")
    print(len(attacks_obj))
    
    return attacks_obj

def load_attack_data(dataset, attack_name, attack_columns, config, return_benign_front=False, return_benign_back=False, evasion_type='cons', **kwargs):

    true_feats = []

    if dataset == 'TEP':
        
        if evasion_type == 'replay_plc' or evasion_type == 'replay_target':
            Xtest, Ytest, sensor_cols = tep_utils.load_tep_attack(dataset, attack_name, spoofing=evasion_type, **kwargs)
        else:
            Xtest, Ytest, sensor_cols = tep_utils.load_tep_attack(dataset, attack_name, **kwargs)

        true_feats.append(tep_utils.attack_footer_to_sensor_idx(attack_name))

        # Labeled start and end
        attack_start = np.min(np.where(Ytest))
        attack_end = np.max(np.where(Ytest))

    elif dataset == 'CTOWN':
        
        attack_number = int(attack_name.split("_")[1])
        
        if evasion_type == 'replay_plc' or evasion_type == 'replay_target':
            Xtest, Ytest, sensor_cols = ctown_utils.load_ctown_attack(dataset, attack_number, spoofing=evasion_type, **kwargs)
        else:
            Xtest, Ytest, sensor_cols = ctown_utils.load_ctown_attack(dataset, attack_number, **kwargs)

        for col_name in attack_columns:
            true_feats.append(sensor_cols.index(col_name))
    
        # Labeled start and end
        attack_start = np.min(np.where(Ytest))
        attack_end = np.max(np.where(Ytest))
    
    elif dataset in ['SWAT', 'WADI']:
        
        attack_idx = int(attack_name.split("_")[1])
        indices, _ = attack_utils.get_attack_indices(dataset)
        
        attack_start = np.min(indices[attack_idx])
        attack_end = np.max(indices[attack_idx]) 

        Xtest, Ytest, sensor_cols = load_test_data(dataset, **kwargs)
        
        for col_name in attack_columns:
            true_feats.append(sensor_cols.index(col_name))
    
    # Where to slice the dataset
    data_start = attack_start - config['history']
    data_end = attack_end + 1

    if return_benign_front and return_benign_back:
        return Xtest, Ytest, true_feats, sensor_cols, attack_start, attack_end    
    
    elif return_benign_front:
        return Xtest[:data_end], Ytest[:data_end], true_feats, sensor_cols, attack_start, attack_end    

    # Should almost never be used, but included for completeness
    elif return_benign_back:
        return Xtest[data_start:], Ytest[data_start:], true_feats, sensor_cols, attack_start, attack_end    

    return Xtest[data_start:data_end], Ytest[data_start:data_end], true_feats, sensor_cols, attack_start, attack_end


def run_detection_eval(device, model, config, dataset='TEP', detection_threshold=None, Xval_true=None, Xval_pred=None, evasion_type='cons', eval_func=run_times_series_test):    
    
    n_features = config['node_num']
    history = config['history']

    attacks_obj = get_attacks_info(dataset, evasion_type)
    full_val_errors = np.abs(Xval_pred - Xval_true)
    per_sample_residuals = np.sum(full_val_errors, axis=1)
    # print("shape of full_val_errors:")
    # print(full_val_errors.shape)
    
    cusum_drift = np.mean(per_sample_residuals) + np.std(per_sample_residuals)
    print("shape of drift before cusum is called: ")
    print(cusum_drift.shape)
    cusum_val = cumulative_sum(Xval_pred, Xval_true, drift=cusum_drift)
    cusum_threshold = np.max(cusum_val) 
    # NOTE: CUSUM threshold can be multiplied by a scaling factor S here. GeCo uses S=2.
    # NOTE: CUSUM threshold can also be capped by a growth factor G here, which limits the CUSUM growth to G * drift. GeCo uses G=5.

    cusum_pf_drift = np.mean(full_val_errors, axis=0) + np.std(full_val_errors, axis=0)
    cusum_pf_val = cumulative_sum(Xval_pred, Xval_true, drift=cusum_pf_drift, per_feature=True)
    cusum_pf_threshold = np.max(cusum_pf_val, axis=0)

    all_cusum = []
    all_pf_cusum = []
    all_labels = []
    all_full_mses = []
    all_mses = []
    all_attack_ids = []

    z_idx = 0

    # NOTE: A bit of a hack; all SWAT attack data is in one file
    if dataset == 'SWAT':
    
        Xtest, Ytest, sensor_cols = load_test_data(dataset)

        test_dataset = TimeSeriesDataset(Xtest.T, Ytest, mode='test', config=config)
        test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
        print("calling eval funct on dataloader:")
        print(len(test_dataloader.dataset))
        _, test_result = eval_func(config["model_name"], model, test_dataloader, sensor_cols)

        print(f'====================** Test Result on all SWAT **=======================\n')

        # Attribution metrics: measure if attacked features are covered
        Xpred, Xtrue, Ytrue = test_result
        print("shape of Ytrue should be 3 here?--", Ytrue.shape)
        full_errors = np.abs(Xtrue - Xpred)

        instance_errors = np.mean(full_errors, axis=1)
        instance_errors = np.convolve(instance_errors, np.ones(3), 'same') / 3
        cusum_test = cumulative_sum(Xpred, Xtrue, drift=cusum_drift)
        cusum_pf_test = cumulative_sum(Xpred, Xtrue, drift=cusum_pf_drift, per_feature=True)

        all_cusum = cusum_test
        all_pf_cusum = cusum_pf_test
        all_labels = Ytrue
        all_mses = instance_errors
        all_full_mses = full_errors
        all_attack_ids = np.zeros_like(instance_errors)

        indices, _ = attack_utils.get_attack_indices(dataset)

        # NOTE: Also hacky, but to match the structure of the other datasets, 
        # we want to segment the entire space by the midpoints between attacks
        attack_midpoints = []
        
        for attack_idx in range(1, len(indices)):

            prev_attack_end = np.max(indices[attack_idx - 1]) - history
            this_attack_start = np.min(indices[attack_idx]) - history

            midpoint = (prev_attack_end + this_attack_start) // 2
            attack_midpoints.append(midpoint)

            print(f'Attack {attack_idx}: {indices[attack_idx][0]} to {indices[attack_idx][-1]}')
            print(f'Midpoint between {attack_idx-1} and {attack_idx} is {midpoint}')

        for i in range(len(attack_midpoints) - 1):
            all_attack_ids[attack_midpoints[i]:attack_midpoints[i+1]] = i+1
        
        all_attack_ids[attack_midpoints[-1]:] = len(attack_midpoints)

        z_idx += 1

    else:
        print("atttack objects length: ", len(attacks_obj))
        # List of tuples
        for attack_name, attack_columns in attacks_obj:

            # Load attack and prep data loader
            Xtest, Ytest, true_feat_idx_list, sensor_cols, _, _ = load_attack_data(dataset, attack_name, attack_columns, config, return_benign_front=True, return_benign_back=False, evasion_type=evasion_type, verbose=0)

            test_dataset = TimeSeriesDataset(Xtest.T, Ytest, mode='test', config=config)
            test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
            _, test_result = eval_func(device, model, test_dataloader, sensor_cols)

            print(f'====================** Test Result {attack_name} on {attack_columns} at true idx: {true_feat_idx_list} **=======================\n')

            # Attribution metrics: measure if attacked features are covered
            Xpred, Xtrue, Ytrue = test_result
            full_errors = np.abs(Xtrue - Xpred)

            if np.any(np.isnan(full_errors)):
                pdb.set_trace()

            instance_errors = np.mean(full_errors, axis=1)
            cusum_test = cumulative_sum(Xpred, Xtrue, drift=cusum_drift)
            cusum_pf_test = cumulative_sum(Xpred, Xtrue, drift=cusum_pf_drift, per_feature=True)

            all_cusum.append(cusum_test)
            all_pf_cusum.append(cusum_pf_test)
            all_labels.append(Ytrue)
            all_mses.append(instance_errors)
            all_full_mses.append(full_errors)
            all_attack_ids.append(np.ones_like(instance_errors) * z_idx)
            z_idx += 1

        all_attack_ids = np.concatenate(all_attack_ids)
        all_cusum = np.concatenate(all_cusum)
        all_pf_cusum = np.concatenate(all_pf_cusum)
        all_mses = np.concatenate(all_mses)
        all_full_mses = np.concatenate(all_full_mses)
        all_labels = np.concatenate(all_labels)

    if detection_threshold is not None:
        
        # Threshold is already given, just report results
        best_threshold = detection_threshold

        for attack_id in range(z_idx):
            
            Ytrue = all_labels[all_attack_ids == attack_id]
            Ypred = all_mses[all_attack_ids == attack_id] > best_threshold

            attack_roc_auc = roc_auc_score(Ytrue, all_mses[all_attack_ids == attack_id])
            
            print('--' * 15)
            print(f'For attack {attack_id}:')
            print('--' * 15)
            print(f'Prec: {precision_score(Ytrue, Ypred):.3f}')
            print(f'Recall: {recall_score(Ytrue, Ypred):.3f}')
            print(f'Point-F1: {f1_score(Ytrue, Ypred):.3f}')
            print(f'ROC AUC: {attack_roc_auc:.3f}')

        final_roc_auc = roc_auc_score(all_labels, all_mses)
        full_pred = all_mses > best_threshold
        full_mask = np.quantile(full_val_errors, 0.9995, axis=0) < all_full_mses
        pf_pred = np.sum(full_mask, axis=1) > 0

        print('--' * 15)
        print(f'Overall')
        print('--' * 15)
        print(f'ROC AUC: {final_roc_auc:.3f}')
        print(f'Total MSE Point-F1: {f1_score(all_labels, full_pred):.3f}')
        print(f'Per Feature Point-F1: {f1_score(all_labels, pf_pred):.3f}')

        all_saved_metrics = dict()
        all_saved_metrics['mse-roc-auc'] = final_roc_auc
        all_saved_metrics['mse-F1'] = f1_score(all_labels, full_pred)
        all_saved_metrics['mse-pf-F1'] = f1_score(all_labels, pf_pred)

    else:

        all_saved_metrics = dict()

        # Do some exploration of different thresholds and their impacts
        print("shape of all_mses: ", all_mses.shape)
        print("shape of all_labels: ", all_labels.shape)    
        fpr, tpr, thresholds = roc_curve(all_labels, all_mses)
        auroc_idx = np.argmax(tpr-fpr) 
        auroc_threshold = thresholds[auroc_idx]
        this_pred = all_mses > auroc_threshold
        print(f'Best AUROC threshold at: {auroc_threshold:.4f}. TPR: {tpr[auroc_idx]:.4f} FPR {fpr[auroc_idx]:.4f}')
        all_saved_metrics['best-auroc-threshold-tpr'] = tpr[auroc_idx]
        all_saved_metrics['best-auroc-threshold-fpr'] = fpr[auroc_idx]
        print_metrics(this_pred, all_labels, all_saved_metrics, 'best-auroc-threshold')
        get_per_attack_metrics(this_pred, all_labels, all_attack_ids, attacks_obj, all_saved_metrics, 'best-auroc-threshold')
        print('--' * 15)

        fpr1_idx = np.where(fpr > 0.01)[0][0]
        fpt1_threshold = thresholds[fpr1_idx]
        this_pred = all_mses > fpt1_threshold
        print(f'FPR 1% threshold at: {fpt1_threshold:.4f}. TPR: {tpr[fpr1_idx]:.4f} FPR {fpr[fpr1_idx]:.4f}')
        all_saved_metrics['fpr1-threshold-tpr'] = tpr[fpr1_idx]
        all_saved_metrics['fpr1-threshold-fpr'] = fpr[fpr1_idx]
        print_metrics(this_pred, all_labels, all_saved_metrics, 'fpr1-threshold')
        get_per_attack_metrics(this_pred, all_labels, all_attack_ids, attacks_obj, all_saved_metrics, 'fpr1-threshold')
        print('--' * 15)

        fpr5_idx = np.where(fpr > 0.05)[0][0]
        fpt5_threshold = thresholds[fpr5_idx]
        this_pred = all_mses > fpt5_threshold
        print(f'FPR 5% threshold at: {fpt5_threshold:.4f}. TPR: {tpr[fpr5_idx]:.4f} FPR {fpr[fpr5_idx]:.4f}')
        all_saved_metrics['fpr5-threshold-tpr'] = tpr[fpr5_idx]
        all_saved_metrics['fpr5-threshold-fpr'] = fpr[fpr5_idx]
        print_metrics(this_pred, all_labels, all_saved_metrics, 'fpr5-threshold')
        get_per_attack_metrics(this_pred, all_labels, all_attack_ids, attacks_obj, all_saved_metrics, 'fpr5-threshold')
        print('--' * 15)

        standard_threshold = np.quantile(np.mean(full_val_errors, axis=1), 0.995)
        this_pred = all_mses > standard_threshold
        print(f'Standard 99.5% validation threshold at: {standard_threshold:.4f}.')      
        print_metrics(this_pred, all_labels, all_saved_metrics, 'mse-val995-threshold')
        get_per_attack_metrics(this_pred, all_labels, all_attack_ids, attacks_obj, all_saved_metrics, 'mse-val995-threshold')
        print('--' * 15)

        maxval_threshold = np.max(np.mean(full_val_errors, axis=1))
        this_pred = all_mses > maxval_threshold
        print(f'Standard max validation threshold at: {maxval_threshold:.4f}.')      
        print_metrics(this_pred, all_labels, all_saved_metrics, 'mse-val1-threshold')
        get_per_attack_metrics(this_pred, all_labels, all_attack_ids, attacks_obj, all_saved_metrics, 'mse-val1-threshold')
        print('--' * 15)

        # NOTE: CUSUM threshold can be multiplied by a scaling factor S here. GeCo uses S=2.
        this_pred = all_cusum > cusum_threshold
        print(f'Standard CUSUM threshold at: {cusum_threshold:.4f}.')
        print_metrics(this_pred, all_labels, all_saved_metrics, 'cusum-total-threshold')
        get_per_attack_metrics(this_pred, all_labels, all_attack_ids, attacks_obj, all_saved_metrics, 'cusum-total-threshold')
        print('--' * 15)

        # NOTE: CUSUM threshold can be multiplied by a scaling factor S here. GeCo uses S=2.
        cusum_test_mask = all_pf_cusum > cusum_pf_threshold 
        this_pred = np.sum(cusum_test_mask, axis=1) > 0
        print(f'With per-feature CUSUM thresholds.')      
        print_metrics(this_pred, all_labels, all_saved_metrics, 'cusum-pf-threshold')
        get_per_attack_metrics(this_pred, all_labels, all_attack_ids, attacks_obj, all_saved_metrics, 'cusum-pf-threshold')
        print('--' * 15)

        # TODO: benchmark AR replay_target, see if something is wrong
        # Need to save: all_pf_cusum, cusum_pf_threshold, this_pred, all_labels, Xpred, Xtrue
        # ff = 'replay_ctown'
        # np.save(f'Xtest_{ff}.npy', Xtest)
        # np.save(f'Xpred_{ff}.npy', Xpred)
        # np.save(f'Xtrue_{ff}.npy', Xtrue)
        # np.save(f'cusum_pf_drift_{ff}.npy', cusum_pf_drift)
        # np.save(f'all_pf_cusum_{ff}.npy', all_pf_cusum)
        # np.save(f'cusum_pf_threshold_{ff}.npy', cusum_pf_threshold)
        # np.save(f'this_pred_{ff}.npy', this_pred)
        # np.save(f'all_labels_{ff}.npy', all_labels)
        # pdb.set_trace()

        full_mask = np.quantile(full_val_errors, 0.995, axis=0) < all_full_mses
        this_pred = np.sum(full_mask, axis=1) > 0
        print(f'With per-feature 99.5 thresholds.')      
        print_metrics(this_pred, all_labels, all_saved_metrics, 'mse-pf-val995-threshold')
        get_per_attack_metrics(this_pred, all_labels, all_attack_ids, attacks_obj, all_saved_metrics, 'mse-pf-val995-threshold')
        print('--' * 15)

        full_mask = np.quantile(full_val_errors, 1.0, axis=0) < all_full_mses
        this_pred = np.sum(full_mask, axis=1) > 0
        print(f'With per-feature max thresholds.')      
        print_metrics(this_pred, all_labels, all_saved_metrics, 'mse-pf-val1-threshold')
        get_per_attack_metrics(this_pred, all_labels, all_attack_ids, attacks_obj, all_saved_metrics, 'mse-pf-val1-threshold')
        print('--' * 15)

        best_threshold = auroc_threshold
        final_roc_auc = roc_auc_score(all_labels, all_mses)
        print(f'ROC AUC: {final_roc_auc:.3f}')
        all_saved_metrics['mse-roc-auc'] = final_roc_auc

        final_cusum_roc_auc = roc_auc_score(all_labels, all_cusum)
        print(f'ROC AUC with CUSUM: {final_cusum_roc_auc:.3f}')
        all_saved_metrics['cusum-roc-auc'] = final_cusum_roc_auc

    return all_saved_metrics, best_threshold


def run_detection_eval_new(device, model, config, dataset='TEP', detection_threshold=None, Xval_true=None, Xval_pred=None, Y_true=None, evasion_type='cons', eval_func=run_times_series_test):    
    
    print("shape of Xval_true and Xval_pred:")
    print(Xval_true.shape, Xval_pred.shape)

    # n_features = config['node_num']
    history = config['history']

    attacks_obj = get_attacks_info(dataset, evasion_type)
    full_val_errors = np.abs(Xval_pred - Xval_true)
    print("shape of full_val_errors:")
    print(full_val_errors.shape)
    per_sample_residuals = np.sum(full_val_errors, axis=1)
    cusum_drift = np.mean(per_sample_residuals) + np.std(per_sample_residuals)
    
    # debug
    print("shape of cusum_drift: ", cusum_drift.shape)
    
    cusum_val = cumulative_sum(Xval_pred, Xval_true, drift=cusum_drift)
    cusum_threshold = np.max(cusum_val) 
    # NOTE: CUSUM threshold can be multiplied by a scaling factor S here. GeCo uses S=2.
    # NOTE: CUSUM threshold can also be capped by a growth factor G here, which limits the CUSUM growth to G * drift. GeCo uses G=5.

    cusum_pf_drift = np.mean(full_val_errors, axis=0) + np.std(full_val_errors, axis=0)
    cusum_pf_val = cumulative_sum(Xval_pred, Xval_true, drift=cusum_pf_drift, per_feature=True)
    cusum_pf_threshold = np.max(cusum_pf_val, axis=0)

    all_cusum = []
    all_pf_cusum = []
    all_labels = []
    all_full_mses = []
    all_mses = []
    all_attack_ids = []

    z_idx = 0

    # NOTE: A bit of a hack; all SWAT attack data is in one file
    if dataset == 'SWAT':
    
        # Xtest, Ytest, sensor_cols = load_test_data(dataset)
        # test_dataset = TimeSeriesDataset(Xtest.T, Ytest, mode='test', config=config)
        # test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
        # _, test_result = eval_func(device, model, test_dataloader, sensor_cols)

        print(f'====================** Test Result on all SWAT **=======================\n')

        # Attribution metrics: measure if attacked features are covered
        # Xpred, Xtrue, Ytrue = test_result
        Xpred, Xtrue = Xval_pred, Xval_true
        
        full_errors = np.abs(Xtrue - Xpred)
        print("Xpred shape:")
        print(Xpred.shape)
        print("Xtrue shape:")
        print(Xtrue.shape)
        instance_errors = np.mean(full_errors, axis=1)
        print("instance_errors before convolv:")
        print(instance_errors.shape)
        instance_errors = np.convolve(instance_errors, np.ones(3), 'same') / 3
        
        print("instance_errors shape:")
        print(instance_errors.shape)
        
        full_val_errors = np.abs(Xval_pred - Xval_true)
        cusum_drift = np.mean(full_val_errors, axis=0) + np.std(full_val_errors, axis=0)
        cusum_test = cumulative_sum(Xpred, Xtrue, drift=cusum_drift)
        cusum_pf_test = cumulative_sum(Xpred, Xtrue, drift=cusum_pf_drift, per_feature=True)
        
        cusum_pf_drift = cusum_drift
        
        all_cusum = cusum_test
        all_pf_cusum = cusum_pf_test
        if Y_true is None:
            print("YTRUE IS NONE")
        all_labels = Y_true
        print("printing Y_true")
        print(Y_true)
        all_mses = instance_errors
        all_full_mses = full_errors
        all_attack_ids = np.zeros_like(instance_errors)

        indices, _ = attack_utils.get_attack_indices(dataset)
        
        # NOTE: Also hacky, but to match the structure of the other datasets, 
        # we want to segment the entire space by the midpoints between attacks
        attack_midpoints = []
        
        for attack_idx in range(1, len(indices)):

            prev_attack_end = np.max(indices[attack_idx - 1]) - history
            this_attack_start = np.min(indices[attack_idx]) - history

            midpoint = (prev_attack_end + this_attack_start) // 2
            attack_midpoints.append(midpoint)

            print(f'Attack {attack_idx}: {indices[attack_idx][0]} to {indices[attack_idx][-1]}')
            print(f'Midpoint between {attack_idx-1} and {attack_idx} is {midpoint}')

        for i in range(len(attack_midpoints) - 1):
            all_attack_ids[attack_midpoints[i]:attack_midpoints[i+1]] = i+1
        
        all_attack_ids[attack_midpoints[-1]:] = len(attack_midpoints)

        z_idx += 1

    else:

        # # List of tuples
        # for attack_name, attack_columns in attacks_obj:
        #     # Load attack and prep data loader
        #     Xtest, Ytest, true_feat_idx_list, sensor_cols, _, _ = load_attack_data(dataset, attack_name, attack_columns, config, return_benign_front=True, return_benign_back=False, evasion_type=evasion_type, verbose=0)

        #     test_dataset = TimeSeriesDataset(Xtest.T, Ytest, mode='test', config=config)
        #     test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
        #     _, test_result = eval_func(device, model, test_dataloader, sensor_cols)

        #     print(f'====================** Test Result {attack_name} on {attack_columns} at true idx: {true_feat_idx_list} **=======================\n')

        #     # Attribution metrics: measure if attacked features are covered
        #     Xpred, Xtrue, Ytrue = test_result
            
        #     full_errors = np.abs(Xtrue - Xpred)

        #     if np.any(np.isnan(full_errors)):
        #         pdb.set_trace()


        #     instance_errors = np.mean(full_errors, axis=1)
        #     cusum_test = cumulative_sum(Xpred, Xtrue, drift=cusum_drift)
        #     cusum_pf_test = cumulative_sum(Xpred, Xtrue, drift=cusum_pf_drift, per_feature=True)

        #     all_cusum.append(cusum_test)
        #     all_pf_cusum.append(cusum_pf_test)
        #     all_labels.append(Ytrue)
        #     all_mses.append(instance_errors)
        #     all_full_mses.append(full_errors)
        #     all_attack_ids.append(np.ones_like(instance_errors) * z_idx)
        #     z_idx += 1

        all_attack_ids = np.concatenate(all_attack_ids)
        all_cusum = np.concatenate(all_cusum)
        all_pf_cusum = np.concatenate(all_pf_cusum)

        all_mses = np.concatenate(all_mses)
        all_full_mses = np.concatenate(all_full_mses)
        all_labels = np.concatenate(all_labels)

    if detection_threshold is not None:
        
        # Threshold is already given, just report results
        best_threshold = detection_threshold

        for attack_id in range(z_idx):
            
            Ytrue = all_labels[all_attack_ids == attack_id]
            Ypred = all_mses[all_attack_ids == attack_id] > best_threshold

            attack_roc_auc = roc_auc_score(Ytrue, all_mses[all_attack_ids == attack_id])
            
            print('--' * 15)
            print(f'For attack {attack_id}:')
            print('--' * 15)
            print(f'Prec: {precision_score(Ytrue, Ypred):.3f}')
            print(f'Recall: {recall_score(Ytrue, Ypred):.3f}')
            print(f'Point-F1: {f1_score(Ytrue, Ypred):.3f}')
            print(f'ROC AUC: {attack_roc_auc:.3f}')

        final_roc_auc = roc_auc_score(all_labels, all_mses)
        full_pred = all_mses > best_threshold
        full_mask = np.quantile(full_val_errors, 0.9995, axis=0) < all_full_mses
        pf_pred = np.sum(full_mask, axis=1) > 0

        print('--' * 15)
        print(f'Overall')
        print('--' * 15)
        print(f'ROC AUC: {final_roc_auc:.3f}')
        print(f'Total MSE Point-F1: {f1_score(all_labels, full_pred):.3f}')
        print(f'Per Feature Point-F1: {f1_score(all_labels, pf_pred):.3f}')

        all_saved_metrics = dict()
        all_saved_metrics['mse-roc-auc'] = final_roc_auc
        all_saved_metrics['mse-F1'] = f1_score(all_labels, full_pred)
        all_saved_metrics['mse-pf-F1'] = f1_score(all_labels, pf_pred)

    else:

        all_saved_metrics = dict()

        # Do some exploration of different thresholds and their impacts
        print("all_labels shape")
        print(all_labels.shape)
        print("mses shape:")
        print(all_mses.shape)
        fpr, tpr, thresholds = roc_curve(all_labels, all_mses)
        auroc_idx = np.argmax(tpr-fpr) 
        auroc_threshold = thresholds[auroc_idx]
        this_pred = all_mses > auroc_threshold
        print(f'Best AUROC threshold at: {auroc_threshold:.4f}. TPR: {tpr[auroc_idx]:.4f} FPR {fpr[auroc_idx]:.4f}')
        all_saved_metrics['best-auroc-threshold-tpr'] = tpr[auroc_idx]
        all_saved_metrics['best-auroc-threshold-fpr'] = fpr[auroc_idx]
        print_metrics(this_pred, all_labels, all_saved_metrics, 'best-auroc-threshold')
        get_per_attack_metrics(this_pred, all_labels, all_attack_ids, attacks_obj, all_saved_metrics, 'best-auroc-threshold')
        print('--' * 15)

        fpr1_idx = np.where(fpr > 0.01)[0][0]
        fpt1_threshold = thresholds[fpr1_idx]
        this_pred = all_mses > fpt1_threshold
        print(f'FPR 1% threshold at: {fpt1_threshold:.4f}. TPR: {tpr[fpr1_idx]:.4f} FPR {fpr[fpr1_idx]:.4f}')
        all_saved_metrics['fpr1-threshold-tpr'] = tpr[fpr1_idx]
        all_saved_metrics['fpr1-threshold-fpr'] = fpr[fpr1_idx]
        print_metrics(this_pred, all_labels, all_saved_metrics, 'fpr1-threshold')
        get_per_attack_metrics(this_pred, all_labels, all_attack_ids, attacks_obj, all_saved_metrics, 'fpr1-threshold')
        print('--' * 15)

        fpr5_idx = np.where(fpr > 0.05)[0][0]
        fpt5_threshold = thresholds[fpr5_idx]
        this_pred = all_mses > fpt5_threshold
        print(f'FPR 5% threshold at: {fpt5_threshold:.4f}. TPR: {tpr[fpr5_idx]:.4f} FPR {fpr[fpr5_idx]:.4f}')
        all_saved_metrics['fpr5-threshold-tpr'] = tpr[fpr5_idx]
        all_saved_metrics['fpr5-threshold-fpr'] = fpr[fpr5_idx]
        print_metrics(this_pred, all_labels, all_saved_metrics, 'fpr5-threshold')
        get_per_attack_metrics(this_pred, all_labels, all_attack_ids, attacks_obj, all_saved_metrics, 'fpr5-threshold')
        print('--' * 15)

        standard_threshold = np.quantile(np.mean(full_val_errors, axis=1), 0.995)
        this_pred = all_mses > standard_threshold
        print(f'Standard 99.5% validation threshold at: {standard_threshold:.4f}.')      
        print_metrics(this_pred, all_labels, all_saved_metrics, 'mse-val995-threshold')
        get_per_attack_metrics(this_pred, all_labels, all_attack_ids, attacks_obj, all_saved_metrics, 'mse-val995-threshold')
        print('--' * 15)

        maxval_threshold = np.max(np.mean(full_val_errors, axis=1))
        this_pred = all_mses > maxval_threshold
        print(f'Standard max validation threshold at: {maxval_threshold:.4f}.')      
        print_metrics(this_pred, all_labels, all_saved_metrics, 'mse-val1-threshold')
        get_per_attack_metrics(this_pred, all_labels, all_attack_ids, attacks_obj, all_saved_metrics, 'mse-val1-threshold')
        print('--' * 15)

        # NOTE: CUSUM threshold can be multiplied by a scaling factor S here. GeCo uses S=2.
        this_pred = all_cusum > cusum_threshold
        print(f'Standard CUSUM threshold at: {cusum_threshold:.4f}.')
        print_metrics(this_pred, all_labels, all_saved_metrics, 'cusum-total-threshold')
        get_per_attack_metrics(this_pred, all_labels, all_attack_ids, attacks_obj, all_saved_metrics, 'cusum-total-threshold')
        print('--' * 15)

        # NOTE: CUSUM threshold can be multiplied by a scaling factor S here. GeCo uses S=2.
        cusum_test_mask = all_pf_cusum > cusum_pf_threshold 
        this_pred = np.sum(cusum_test_mask, axis=1) > 0
        print(f'With per-feature CUSUM thresholds.')      
        print_metrics(this_pred, all_labels, all_saved_metrics, 'cusum-pf-threshold')
        get_per_attack_metrics(this_pred, all_labels, all_attack_ids, attacks_obj, all_saved_metrics, 'cusum-pf-threshold')
        print('--' * 15)

        full_mask = np.quantile(full_val_errors, 0.995, axis=0) < all_full_mses
        this_pred = np.sum(full_mask, axis=1) > 0
        print(f'With per-feature 99.5 thresholds.')      
        print_metrics(this_pred, all_labels, all_saved_metrics, 'mse-pf-val995-threshold')
        get_per_attack_metrics(this_pred, all_labels, all_attack_ids, attacks_obj, all_saved_metrics, 'mse-pf-val995-threshold')
        print('--' * 15)

        full_mask = np.quantile(full_val_errors, 1.0, axis=0) < all_full_mses
        this_pred = np.sum(full_mask, axis=1) > 0
        print(f'With per-feature max thresholds.')      
        print_metrics(this_pred, all_labels, all_saved_metrics, 'mse-pf-val1-threshold')
        get_per_attack_metrics(this_pred, all_labels, all_attack_ids, attacks_obj, all_saved_metrics, 'mse-pf-val1-threshold')
        print('--' * 15)

        best_threshold = auroc_threshold
        final_roc_auc = roc_auc_score(all_labels, all_mses)
        print(f'ROC AUC: {final_roc_auc:.3f}')
        all_saved_metrics['mse-roc-auc'] = final_roc_auc

        final_cusum_roc_auc = roc_auc_score(all_labels, all_cusum)
        print(f'ROC AUC with CUSUM: {final_cusum_roc_auc:.3f}')
        all_saved_metrics['cusum-roc-auc'] = final_cusum_roc_auc

    return all_saved_metrics, best_threshold

def get_per_attack_metrics(pred, labels, all_attack_ids, attacks_obj, save_dict, save_dict_header):

    per_attack_results = dict()

    for z_idx in range(len(attacks_obj)):
    
        attack_name, attack_columns = attacks_obj[z_idx]
        per_attack_results[attack_name] = dict()

        attack_pred = pred[all_attack_ids == z_idx]
        attack_labels = labels[all_attack_ids == z_idx]
        
        in_attack_preds = attack_pred[attack_labels == 1]
        
        # TODO: Consider handling special cases where the prediction starts with a FP?
        # These will be captured by the overall precision numbers
        if np.any(in_attack_preds):
            # Get the first detection
            detect_start = np.where(in_attack_preds)[0][0]
            recall = np.mean(in_attack_preds)
        else:
            detect_start = -1
            recall = np.mean(in_attack_preds)

        per_attack_results[attack_name]['detect_start'] = int(detect_start)
        per_attack_results[attack_name]['recall'] = recall
        print(f'{save_dict_header} {attack_name}: detect t={detect_start}, recall = {recall}')

    save_dict[f'{save_dict_header}-per-attack'] = per_attack_results

def print_metrics(pred, labels, save_dict, save_dict_header):

    for metric_name in ['precision', 'recall', 'F1', 'FP', 'TP', 'SF1', 'TAF1', 'NAB']:
        
        metric_func = metrics.get(metric_name)
        final_score = metric_func(pred, labels)
        save_dict[f'{save_dict_header}-{metric_name}'] = final_score
        print(f'{metric_name}: {final_score}')

def run_attribution_eval_true_timing(device, model, config, metrics_obj, dataset='TEP', evasion_type='cons', eval_func=run_test):    
    
    n_features = config['node_num']
    history = config['history']
    attacks_obj = get_attacks_info(dataset, evasion_type)
    
    first_timing_ranks = np.zeros(len(attacks_obj))
    first_timing_sub_ranks = np.zeros(len(attacks_obj))
    first_timing_plc_ranks = np.zeros(len(attacks_obj))
    first_timing_analysis_mses = np.zeros((len(attacks_obj), n_features))
    
    ideal_timing_ranks = np.zeros(len(attacks_obj))
    ideal_timing_sub_ranks = np.zeros(len(attacks_obj))
    ideal_timing_plc_ranks = np.zeros(len(attacks_obj))
    ideal_timing_analysis_mses = np.zeros((len(attacks_obj), n_features))

    z_idx = 0

    # List of tuples
    for attack_name, attack_columns in attacks_obj:

        # Load attack and prep data loader
        Xtest, Ytest, true_feat_idx_list, sensor_cols, attack_start, _ = load_attack_data(dataset, attack_name, attack_columns, config, evasion_type=evasion_type)

        test_dataset = TimeSeriesDataset(Xtest.T, Ytest, mode='test', config=config)
        test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
        _, test_result = eval_func(device, model, test_dataloader, sensor_cols)

        print(f'====================** Test Result {attack_name} on {attack_columns} at true idx: {true_feat_idx_list} **=======================\n')

        # Attribution metrics: measure if attacked features are covered
        Xpred, Xtrue, Ytrue = test_result
        full_errors = np.abs(Xtrue - Xpred)

        #########################################
        # By true timing
        first_timing_mse = full_errors[0]
        first_timing_ranks[z_idx] = attack_utils.scores_to_rank(first_timing_mse, true_feat_idx_list)
        first_timing_analysis_mses[z_idx] = first_timing_mse

        # By ideal timing (shifted by history)
        ideal_timing_mse = full_errors[history]
        ideal_timing_ranks[z_idx] = attack_utils.scores_to_rank(ideal_timing_mse, true_feat_idx_list)
        ideal_timing_analysis_mses[z_idx] = ideal_timing_mse

        # subsystems_ranks
        first_timing_sub_ranks[z_idx] = attack_utils.scores_to_subsystem_rank(dataset, first_timing_mse, true_feat_idx_list)
        ideal_timing_sub_ranks[z_idx] = attack_utils.scores_to_subsystem_rank(dataset, ideal_timing_mse, true_feat_idx_list)

        first_timing_plc_ranks[z_idx] = attack_utils.scores_to_subsystem_rank(dataset, first_timing_mse, true_feat_idx_list, use_plcs=True)
        ideal_timing_plc_ranks[z_idx] = attack_utils.scores_to_subsystem_rank(dataset, ideal_timing_mse, true_feat_idx_list, use_plcs=True)

        z_idx += 1

    print('-' * 15)
    print(f'AvgRank: first timing: {np.mean(first_timing_ranks) / n_features} ideal timing: {np.mean(ideal_timing_ranks) / n_features}')
    print(f'Top-3 Matches: first timing: {np.sum(first_timing_ranks < 4)} / {len(first_timing_ranks)} ideal timing: {np.sum(ideal_timing_ranks < 4)} / {len(ideal_timing_ranks)}')
    print('-' * 15)
    print(f'How many features were searched? first timing: {np.mean(first_timing_ranks)} ideal timing: {np.mean(ideal_timing_ranks)}')
    print('-' * 15)

    metrics_obj['first-timing-avgrank'] = np.mean(first_timing_ranks)
    metrics_obj['ideal-timing-avgrank'] = np.mean(ideal_timing_ranks)
    metrics_obj['first-timing-subrank'] = np.mean(first_timing_sub_ranks)
    metrics_obj['ideal-timing-subrank'] = np.mean(ideal_timing_sub_ranks)
    metrics_obj['first-timing-plcrank'] = np.mean(first_timing_plc_ranks)
    metrics_obj['ideal-timing-plcrank'] = np.mean(ideal_timing_plc_ranks)

    return np.mean(first_timing_ranks), np.mean(ideal_timing_ranks)

def run_attribution_eval_detections(device, model, config, metrics_obj, dataset='TEP', evasion_type='cons', eval_func=run_test):    

    n_features = config['node_num']
    history = config['history']
    attacks_obj = get_attacks_info(dataset, evasion_type)

    # List of tuples
    for attack_name, attack_columns in attacks_obj:

        # Load attack and prep data loader
        Xtest, Ytest, true_feat_idx_list, sensor_cols, attack_start, _ = load_attack_data(dataset, attack_name, attack_columns, config, evasion_type=evasion_type)

        test_dataset = TimeSeriesDataset(Xtest.T, Ytest, mode='test', config=config)
        test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
        _, test_result = eval_func(device, model, test_dataloader, sensor_cols)

        print(f'====================** Test Result {attack_name} on {attack_columns} at true idx: {true_feat_idx_list} **=======================\n')

        # Attribution metrics: measure if attacked features are covered
        Xpred, Xtrue, Ytrue = test_result
        full_errors = np.abs(Xtrue - Xpred)

        for threshold in LIST_OF_THRESHOLDS:

            key = f'{threshold}-per-attack'

            # Get the detection time for this detection threshold
            first_alarm_idx = metrics_obj[key][attack_name]['detect_start']
            
            if first_alarm_idx >= 0:
                
                detection_mse = full_errors[first_alarm_idx]
                
                rank = attack_utils.scores_to_rank(detection_mse, true_feat_idx_list)
                metrics_obj[key][attack_name]['detection_rank'] = int(rank)
                
                sub_rank = attack_utils.scores_to_subsystem_rank(dataset, detection_mse, true_feat_idx_list)
                metrics_obj[key][attack_name]['detection_subrank'] = int(sub_rank)

                plc_rank = attack_utils.scores_to_subsystem_rank(dataset, detection_mse, true_feat_idx_list, use_plcs=True)
                metrics_obj[key][attack_name]['detection_plcrank'] = int(plc_rank)

            else:
                metrics_obj[key][attack_name]['detection_rank'] = -1
                metrics_obj[key][attack_name]['detection_subrank'] = -1
                metrics_obj[key][attack_name]['detection_plcrank'] = -1

    print('-' * 15)
    print(f'How many features were searched?') 
    
    for threshold in LIST_OF_THRESHOLDS:
    
        key = f'{threshold}-per-attack'
        per_attack_outcomes = metrics_obj[key]
        detect_ranks = np.zeros(len(per_attack_outcomes))
        z_idx = 0

        for attack_name in per_attack_outcomes:
            
            rank = per_attack_outcomes[attack_name]['detection_rank']
            detect_ranks[z_idx] = rank
            z_idx += 1

        print(f'{key}: detection avgrank {np.mean(detect_ranks[detect_ranks > 0])}')
        metrics_obj[f'{threshold}-detection-timing-avgrank'] = np.mean(detect_ranks[detect_ranks > 0])

    print('-' * 15)

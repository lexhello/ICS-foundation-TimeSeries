import pdb
import pickle
import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader, Subset

def get_loaders(train_dataset, batch, val_ratio=0.1):
    dataset_len = int(len(train_dataset))
    
    train_use_len = int(dataset_len * (1 - val_ratio)) 

    indices = torch.arange(dataset_len)
    train_subset = Subset(train_dataset, indices[:train_use_len])
    val_subset = Subset(train_dataset, indices[train_use_len:])

    train_dataloader = DataLoader(train_subset, batch_size=batch,
                            shuffle=False)

    val_dataloader = DataLoader(val_subset, batch_size=batch,
                            shuffle=False)

    return train_dataloader, val_dataloader

class TimeSeriesDataset(Dataset):

    def __init__(self, raw_data, labels, mode='train', config = None):
        
        self.raw_data = raw_data
        self.config = config
        self.mode = mode

        # to tensor
        data = torch.tensor(raw_data).double()
        labels = torch.tensor(labels).double()

        self.x, self.y, self.labels = self.process(data, labels)
    
    def __len__(self):
        return len(self.x)

    def process(self, data, labels):
        x_arr, y_arr = [], []
        labels_arr = []

        slide_win, slide_stride = [self.config[k] for k
            in ['history', 'slide_stride']
        ]
        is_train = self.mode == 'train'

        node_num, total_time_len = data.shape

        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)
        
        for i in rang:

            ft = data[:, i-slide_win:i]
            tar = data[:, i]

            x_arr.append(ft)
            y_arr.append(tar)

            labels_arr.append(labels[i])

        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()

        labels = torch.Tensor(labels_arr).contiguous()
        
        return x, y, labels

    def __getitem__(self, idx):

        feature = self.x[idx].double()
        y = self.y[idx].double()
        label = self.labels[idx].double()

        return feature, y, label

def load_train_data(dataset_name, scaler=None, no_transform=False, verbose=False):

    if verbose:
        print('Loading {} train data...'.format(dataset_name))

    if scaler is None:
        if verbose:
            print("No scaler provided. Using default sklearn.preprocessing.StandardScaler")
        scaler = StandardScaler()

    if dataset_name == 'TEP':
        
        df_train = pd.read_csv("data/TEP/TEP_train.csv", dayfirst=True)
        sensor_cols = [col for col in df_train.columns if col not in ['Atk']]
    
    elif dataset_name == 'CTOWN':
        
        df_train = pd.read_csv("data/CTOWN/CTOWN_train.csv")
        sensor_cols = [col for col in df_train.columns if col not in ['iteration', 'timestamp']]

    elif dataset_name == 'SWAT':
        
        df_train = pd.read_csv("data/" + dataset_name + "/SWATv0_train.csv", dayfirst=True)
        sensor_cols = [col for col in df_train.columns if col not in ['AIT201', 'Timestamp', 'Normal/Attack']]

    elif dataset_name == 'WADI':

        df_train = pd.read_csv("data/" + dataset_name + "/WADI_train.csv")
        remove_list = ['Row', 'Date', 'Time', 'Attack', '2B_AIT_002_PV', '2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS', 'LEAK_DIFF_PRESSURE', 'PLANT_START_STOP_LOG', 'TOTAL_CONS_REQUIRED_FLOW']
        sensor_cols = [col for col in df_train.columns if col not in remove_list]
        #["2_P_001_STATUS", "2_P_002_STATUS", "2_LS_002_AL", "2_LS_001_AL", "2B_AIT_002_PV", "2_PIC_003_SP", "1_MV_002_STATUS", "1_MV_003_STATUS"]

    else:
        raise ValueError('Dataset name not found.')
        
    # scale sensor data
    if no_transform:
        X = pd.DataFrame(index = df_train.index, columns = sensor_cols, data = df_train[sensor_cols].values)
    else:
        X_prescaled = df_train[sensor_cols].values
        X = pd.DataFrame(index = df_train.index, columns = sensor_cols, data = scaler.fit_transform(X_prescaled))

        # Need fitted scaler for future attack/test data        
        pickle.dump(scaler, open(f'checkpoints/{dataset_name}_scaler.pkl', 'wb'))
        if verbose:
            print('Saved scaler parameters to {}.'.format('scaler.pkl'))

    return X.values, sensor_cols

def load_test_data(dataset_name, scaler=None, no_transform=False, verbose=False):

    if verbose:
        print('Loading {} test data...'.format(dataset_name))

    if scaler is None:
        print('No scaler provided, trying to load from checkpoints directory...')
        scaler = pickle.load(open(f'checkpoints/{dataset_name}_scaler.pkl', "rb"))

    if dataset_name == 'TEP':
        
        df_test = pd.read_csv("data/" + dataset_name + "/TEP_test.csv", dayfirst=True)
        sensor_cols = [col for col in df_test.columns if col not in ['Atk']]
        target_col = 'Atk'

    elif dataset_name == 'CTOWN':
        
        df_test = pd.read_csv("data/CTOWN/CTOWN_test.csv")
        sensor_cols = [col for col in df_test.columns if col not in ['iteration', 'timestamp', 'Attack']]
        target_col = 'Attack'

    elif dataset_name == 'SWAT':
        df_test = pd.read_csv("data/" + dataset_name + "/SWATv0_test.csv")
        sensor_cols = [col.strip() for col in df_test.columns if col not in ['AIT201', 'Timestamp', 'Normal/Attack']]
        target_col = 'Normal/Attack'

    elif dataset_name == 'WADI':
        
        df_test = pd.read_csv("data/" + dataset_name + "/WADI_test.csv")
        
        # Remove nan columns
        remove_list = ['Row', 'Date', 'Time', 'Attack', '2B_AIT_002_PV', '2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS', 'LEAK_DIFF_PRESSURE', 'PLANT_START_STOP_LOG', 'TOTAL_CONS_REQUIRED_FLOW']
        sensor_cols = [col for col in df_test.columns if col not in remove_list]
        target_col = 'Attack'

    else:
        raise ValueError('Dataset name not found.')

    if verbose:
        print(f'Loading test data for {dataset_name} successful.')

    # scale sensor data
    if no_transform:
        Xtest = pd.DataFrame(index = df_test.index, columns = sensor_cols, data = df_test[sensor_cols])
    else:
        Xtest = pd.DataFrame(index = df_test.index, columns = sensor_cols, data = scaler.transform(df_test[sensor_cols]))
    
    Ytest = df_test[target_col]

    return Xtest.values, Ytest.values, sensor_cols

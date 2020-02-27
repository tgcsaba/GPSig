import sys
import os
sys.path.append('../..')
import numpy as np
import gpsig

from scipy.io import loadmat

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

def load_dataset(dataset_name, for_model='sig', normalize_data=False, add_time=False, max_len=None, val_split=None, test_split=None, return_min_len=False):
    
    # if test_split is not None it will instead return test_split % of the training data for testing

    data_path = './datasets/{}.mat'.format(dataset_name)
   
    if not os.path.exists(data_path):
        raise ValueError('Please download the attached datasets and extract to the /benchmarks/datasets/ directory...')
        
    data = loadmat(data_path)
    
    X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
    
    X_train, y_train, X_test, y_test = np.squeeze(X_train), np.squeeze(y_train), np.squeeze(X_test), np.squeeze(y_test)
    
    len_min = min(np.min([x.shape[0] for x in X_train]), np.min([x.shape[0] for x in X_test]))
    
    num_train = len(X_train)
    num_test = len(X_test)
    
    num_features = X_train[0].shape[1]
        
    if add_time:
        X_train = gpsig.preprocessing.add_time_to_list(X_train)
        X_test = gpsig.preprocessing.add_time_to_list(X_test)        
        num_features += 1
        
    if max_len is not None:
        # perform mean-pooling of every n subsequent observations such that the length of each sequence <= max_len
        X_train = [x if x.shape[0] <= max_len else
                    np.stack([x[i*int(np.ceil(x.shape[0]/max_len)):np.minimum((i+1)*int(np.ceil(x.shape[0]/max_len)), x.shape[0])].mean(axis=0)
                                for i in range(int(np.ceil(x.shape[0]/np.ceil(x.shape[0]/max_len))))], axis=0) for x in X_train]
        X_test = [x if x.shape[0] <= max_len else
                    np.stack([x[i*int(np.ceil(x.shape[0]/max_len)):np.minimum((i+1)*int(np.ceil(x.shape[0]/max_len)), x.shape[0])].mean(axis=0)
                            for i in range(int(np.ceil(x.shape[0]/np.ceil(x.shape[0]/max_len))))], axis=0) for x in X_test]

    num_classes = np.unique(np.int32(y_train)).size
    
    if val_split is not None:
        if val_split < 1. and np.ceil(val_split * num_train) < 2 * num_classes:
            val_split = 2 * num_classes
        elif val_split > 1. and val_split < 2 * num_classes:
            val_split = 2 * num_classes
    
    if test_split is not None:
        if test_split < 1. and np.ceil(test_split * num_train) < 2 * num_classes:
            test_split = 2 * num_classes
        elif test_split > 1. and test_split < 2 * num_classes:
            test_split = 2 * num_classes
    
    if val_split is not None and test_split is not None:
        if val_split < 1. and test_split > 1:
            val_split = int(np.ceil(num_train * val_split))
        elif val_split > 1 and test_split < 1.:
            test_split = int(np.ceil(num_train * test_split))
                
    split_from_train = val_split + test_split if val_split is not None and test_split is not None else val_split or test_split 

    if split_from_train is not None:

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=split_from_train, shuffle=True, stratify=y_train)
        
        if val_split is not None and test_split is not None:
            X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=float(test_split)/split_from_train, shuffle=True, stratify=y_val)
            num_val = len(X_val)
            num_test = len(X_test)
        elif val_split is not None:
            num_val = len(X_val)
        else:
            X_test, y_test = X_val, y_val
            X_val, y_val = None, None
            num_test = len(X_test)
            num_val = 0
        num_train = len(X_train)
    else:
        X_val, y_val = None, None
        num_val = 0

    if normalize_data:
        scaler = StandardScaler()
        scaler.fit(np.concatenate(X_train, axis=0))
        X_train = [scaler.transform(x) for x in X_train]
        X_val = [scaler.transform(x) for x in X_val] if X_val is not None else None
        X_test = [scaler.transform(x) for x in X_test]
    
    for_model = for_model.lower()
    if X_val is None:
        if for_model.lower() == 'sig':
            X = gpsig.preprocessing.tabulate_list_of_sequences(X_train + X_test)
        elif for_model.lower() == 'nn':
            X = gpsig.preprocessing.tabulate_list_of_sequences(X_train + X_test, pre=True, pad_with=0.)
        elif for_model.lower() == 'kconv':
            X = gpsig.preprocessing.tabulate_list_of_sequences(X_train + X_test, pad_with=float('nan'))
        else:
            raise ValueError('unknown architecture: {}'.format(for_model))
        X_train = X[:num_train]
        X_test = X[num_train:]
    else:
        if for_model.lower() == 'sig':
            X = gpsig.preprocessing.tabulate_list_of_sequences(X_train + X_val + X_test)
        elif for_model.lower() == 'nn':
            X = gpsig.preprocessing.tabulate_list_of_sequences(X_train + X_val + X_test, pre=True, pad_with=0.)
        elif for_model.lower() == 'kconv':
            X = gpsig.preprocessing.tabulate_list_of_sequences(X_train + X_val + X_test, pad_with=float('nan'))
        else:
            raise ValueError('unknown architecture: {}'.format(for_model))
        X_train = X[:num_train]
        X_val = X[num_train:num_train+num_val]
        X_test = X[num_train+num_val:]
    
    labels = {y : i for i, y in enumerate(np.unique(y_train))}

    y_train = np.asarray([labels[y] for y in y_train])
    y_val = np.asarray([labels[y] for y in y_val]) if y_val is not None else None
    y_test = np.asarray([labels[y] for y in y_test])
    
    if return_min_len:
        return X_train, y_train, X_val, y_val, X_test, y_test, len_min
    else:
        return X_train, y_train, X_val, y_val, X_test, y_test
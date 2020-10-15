import sys
sys.path.append('..')

from gpsig.preprocessing import pad_list_of_sequences, interp_list_of_sequences, add_time_to_list

import numpy as np
import requests
import zipfile
import io
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

import time
from tqdm import tqdm_notebook as tqdm

def quadratic_time_mmd(K_XX, K_XY, K_YY):
    n = len(K_XX)
    m = len(K_YY)
    # Unbiased MMD statistic
    mmd = (np.sum(K_XX) - np.sum(np.diag(K_XX))) / (n*(n-1))  + (np.sum(K_YY) - np.sum(np.diag(K_YY))) / (m*(m-1))  - 2*np.sum(K_XY)/(n*m)
    return mmd

def mmd_optimized(compute_kernel_matrices, X, Y, params_grid):
    """
    Parameters
    ----------
    
    X : (m_samples, dim) numpy array
    Y : (n_samples, dim) numpy array
    kernel: fuction that evaluates X,Y,g to a matrix

    Returns
    -------
    (m_samples,n_samples) numpy array
    
    """
    times = []
    
    d_opt = -1.0
    p_opt = 0
    
    for i, p in tqdm(enumerate(params_grid), total=len(params_grid)): 
        st = time.time()
        
        K_XX, K_XY, K_YY = normalize_kernel_matrices(*compute_kernel_matrices(X, Y, **p))
        
        d_new= quadratic_time_mmd(K_XX, K_XY, K_YY)
        
        if d_new > d_opt:
            d_opt = d_new
            p_opt = p
        times.append(time.time() - st)
    
    t_sum = np.sum(times)
    t_avg = np.mean(times)
    
    return d_opt, p_opt, t_sum, t_avg

def mmd_optimized_with_summary(kernel, X, Y, params_grid, name=None):
    print(f'Kernel: {name}')
    print('------------------')
    
    d, p, t_sum, t_avg = mmd_optimized(kernel, X, Y, params_grid)
    
    print(f'Time elapsed: {t_sum:0.2f}')
    print(f'Time per iteration: {t_avg:0.2f}')
    print(f'Number of combinations: {len(params_grid)}')
    print(f'Found parameters: {p}')
    print(f'MMD: {d:0.3e}')
    print('------------------')
    print()
    return d

def mmd_with_sklearn_kernel(kernel, X, Y, params_grid, name=None):
    X, Y, _, _ = preprocess_data_for_vector_kernel(X, Y)
    def compute_kernel_matrices(X, Y, **kwargs):
        return kernel(X, X, **kwargs), kernel(X, Y, **kwargs), kernel(Y, Y, **kwargs)
    d = mmd_optimized_with_summary(compute_kernel_matrices, X, Y, params_grid, name=name)
    return d

def mmd_with_sklearn_gp_kernel(kernel, X, Y, params_grid, name=None):
    X, Y, _, _ = preprocess_data_for_vector_kernel(X, Y)
    def compute_kernel_matrices(X, Y, **kwargs):
        kernel_fn = kernel(**kwargs)
        return kernel_fn(X, X), kernel_fn(X, Y), kernel_fn(Y, Y)
    d = mmd_optimized_with_summary(compute_kernel_matrices, X, Y, params_grid, name=name)
    return d

def mmd_with_gpflow_kernel(kernel, X, Y, params_grid, name=None):
    def compute_kernel_matrices(X, Y, **kwargs):
        return compute_gpflow_kernel_matrices(kernel, X, Y, **kwargs)
    d = mmd_optimized_with_summary(compute_kernel_matrices, X, Y, params_grid, name=name)
    return d

def mmd_with_gpsig_kernel(kernel, X, Y, params_grid, low_rank=False, batch_size=50, name=None):
    def compute_kernel_matrices(X, Y, **kwargs):
        return compute_gpsig_kernel_matrices(kernel, X, Y, low_rank=low_rank, batch_size=batch_size, **kwargs)
    d = mmd_optimized_with_summary(compute_kernel_matrices, X, Y, params_grid, name=name)
    return d

def load_netflow_dataset(normalize=True):
    dataset_name = 'NetFlow'
    dataset_url = 'https://www.dropbox.com/s/y25383d8mxq9spa/NetFlow.zip?dl=1'

    # download, extract and load dataset
    r = requests.get(dataset_url, stream=True)
    if r.ok:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall('./{}/'.format(dataset_name))
    else:
        print('Error: the dataset could not be downloaded.')
            
    data = loadmat('./{0}/{0}.mat'.format('NetFlow'))    
    return data

def load_japvowels_dataset(normalize=True):
    dataset_name = 'JapaneseVowels'
    dataset_url = 'https://www.dropbox.com/s/5psb6surmeboq09/JapaneseVowels.zip?dl=1'

    # download, extract and load dataset
    r = requests.get(dataset_url, stream=True)
    if r.ok:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall('./{}/'.format(dataset_name))
    else:
        print('Error: the dataset could not be downloaded.')
            
    data = loadmat('./{0}/{0}.mat'.format(dataset_name))    
    return data

def normalize_data(X):
    scaler = StandardScaler()
    scaler.fit(np.concatenate(X, axis=0))
    X = [scaler.transform(x) for x in X]
    return X

def preprocess_data_for_sig_kernel(X, Y, normalize=True):
    X = list(X)
    Y = list(Y)
    if normalize:
        X = normalize_data(X)
        Y = normalize_data(Y)
    num_X = len(X)
    num_Y = len(Y)
    
    X, Y = np.split(pad_list_of_sequences(X + Y), [num_X])
    
    len_examples = X.shape[1]
    num_features = X.shape[2]
    
    X = np.reshape(X, [num_X, -1])
    Y = np.reshape(Y, [num_Y, -1])
    
    return X, Y, len_examples, num_features

def preprocess_data_for_vector_kernel(X, Y, normalize=True, tabulation='interp'):
    X = list(X)
    Y = list(Y)
    
    if normalize:
        X = normalize_data(X)
        Y = normalize_data(Y)
    
    num_X = len(X)
    num_Y = len(Y)
    
    if tabulation == 'interp':
        X, Y = np.split(interp_list_of_sequences(X + Y), [num_X])
    elif tabulation == 'pad':
        X, Y = np.split(pad_list_of_sequences(X + Y, pad_with=0.), [num_X])
        
    len_examples = X.shape[1]
    num_features = X.shape[2]
    
    X = np.reshape(X, [num_X, -1])
    Y = np.reshape(Y, [num_Y, -1])
    
    return X, Y, len_examples, num_features

def normalize_kernel_matrices(K_XX, K_XY, K_YY, jitter=1e-6):
    K_XX += jitter
    K_XY += jitter
    K_YY += jitter
    
    K_X = np.diag(K_XX)
    K_Y = np.diag(K_YY)
    
    K_XY /= np.sqrt(K_X[:, None] * K_Y[None, :])
    K_XX /= np.sqrt(K_X[:, None] * K_X[None, :])
    K_YY /= np.sqrt(K_Y[:, None] * K_Y[None, :])
    
    return K_XX, K_XY, K_YY
    

def compute_kernel_in_batches(kernel_fn, X, Y, batch_size=50):
    num_X, num_Y = X.shape[0], Y.shape[0]
    K = np.zeros((num_X, num_Y))
    for i in range(int(np.ceil(float(num_X)/batch_size))):
        lower_X, upper_X = i*batch_size, np.min((i+1)*batch_size)
        batch_X = X[lower_X:upper_X]
        for j in range(int(np.ceil(float(num_Y)/batch_size))):
            lower_Y, upper_Y = j*batch_size, np.min((j+1)*batch_size)
            batch_Y = Y[lower_Y:upper_Y]
            K[lower_X:upper_X, lower_Y:upper_Y] = kernel_fn(batch_X, batch_Y)
    return K

def compute_feature_in_batches(feature_fn, X, batch_size=50):
    num_X = X.shape[0]
    P = []
    for i in range(int(np.ceil(float(num_X)/batch_size))):
        lower_X, upper_X = i*batch_size, np.min((i+1)*batch_size)
        batch_X = X[lower_X:upper_X]
        P.append(feature_fn(batch_X))
    P = np.concatenate(P, axis=0)
    return P

def compute_gpflow_kernel_matrices(kernel, X, Y, tabulation='interp', add_time=False, batch_size=None, **kwargs):
    
    if add_time:
        X = add_time_to_list(X)
        Y = add_time_to_list(Y)
    
    X, Y, _, _ = preprocess_data_for_vector_kernel(X, Y, tabulation==tabulation)
                
    
    k = kernel(**kwargs)
    kernel_fn = lambda X, Y: k.K(X, Y).numpy()
    if batch_size is None:
        K_XX = kernel_fn(X, X)
        K_XY = kernel_fn(X, Y)
        K_YY = kernel_fn(Y, Y)
    else:
        K_XX = compute_kernel_in_batches(kernel_fn, X, X)
        K_XY = compute_kernel_in_batches(kernel_fn, X, Y)
        K_YY = compute_kernel_in_batches(kernel_fn, Y, Y)
        
    return K_XX, K_XY, K_YY

def compute_gpsig_kernel_matrices(kernel, X, Y, low_rank=False, add_time=True, batch_size=None, **kwargs):
    
    if add_time:
        X = add_time_to_list(X)
        Y = add_time_to_list(Y)
    
    X, Y, len_examples, num_features = preprocess_data_for_sig_kernel(X, Y)
    
    if not low_rank:
        k = kernel(num_features, len_examples, normalization=True, **kwargs)
        kernel_fn = lambda X, Y: k.K(X, Y).numpy()
        K_XX = compute_kernel_in_batches(kernel_fn, X, X)
        K_XY = compute_kernel_in_batches(kernel_fn, X, Y)
        K_YY = compute_kernel_in_batches(kernel_fn, Y, Y)
    else:
        k = kernel(num_features, len_examples, normalization=True, low_rank=True, **kwargs)
        k.fit_low_rank_params(np.concatenate((X, Y), axis=0))
        feature_fn = lambda X: k.feat(X).numpy()
        P_X = compute_feature_in_batches(feature_fn, X, batch_size=batch_size)
        P_Y = compute_feature_in_batches(feature_fn, Y, batch_size=batch_size)
        K_XX = P_X @ P_X.T
        K_XY = P_X @ P_Y.T
        K_YY = P_Y @ P_Y.T
        
    return K_XX, K_XY, K_YY
import sys
sys.path.append('..')

import numpy as np
from sklearn.preprocessing import StandardScaler
 
from .kernels import SignatureKernel
from gpflow.kernels import Kernel as GPFlowKernel
from sklearn.gaussian_process.kernels import Kernel as SkGPKernel

from gpsig.preprocessing import pad_list_of_sequences, interp_list_of_sequences, add_time_to_list, add_time_to_table

from functools import partial

import time

try:
    from IPython import get_ipython
    if 'IPKernelApp' not in get_ipython().config:
        raise ImportError("console")
except:
    from tqdm import tqdm as tqdm
else:
    from tqdm import tqdm_notebook as tqdm


### Data preprocessing ##

def normalize_data(X, Y=None):
    scaler = StandardScaler()
    if Y is not None:
        scaler.fit(np.concatenate(X+Y, axis=0))
        X = [scaler.transform(x) for x in X]
        Y = [scaler.transform(x) for x in Y]
        return X, Y
    else:
        scaler.fit(np.concatenate(X, axis=0))
        X = [scaler.transform(x) for x in X]
        return X

# def preprocess_data(X, Y=None, normalize=False):
#     X = list(X)
#     num_X = len(X)
#     if Y is not None:
#         Y = list(Y)
#         if normalize:
#             X, Y = normalize_data(X, Y)
#         num_Y = len(Y)
#         X, Y = np.split(pad_list_of_sequences(X + Y), [num_X])
#     else:
#         if normalize:
#             X = normalize_data(X)
#         X = pad_list_of_sequences(X)
    
#     len_examples = X.shape[1]
#     num_features = X.shape[2]
    
#     X = np.reshape(X, [num_X, -1])
#     if Y is not None:
#         Y = np.reshape(Y, [num_Y, -1])
#         return X, Y, len_examples, num_features
#     else:
#         return X, len_examples, num_features

def preprocess_data(X, Y=None, normalize=False, tabulation=None):
    X = list(X)
    num_X = len(X)
    
    if Y is not None:
        Y = list(Y)
        if normalize:
            X, Y = normalize_data(X, Y)
        num_Y = len(Y)
        if tabulation == 'interp':
            X, Y = np.split(interp_list_of_sequences(X + Y), [num_X])
        elif tabulation == 'obspad':
            X, Y = np.split(pad_list_of_sequences(X + Y), [num_X])
        elif tabulation == 'zeropad':
            X, Y = np.split(pad_list_of_sequences(X + Y, pad_with=0.), [num_X])
        X, Y = np.asarray(X), np.asarray(Y)
    else:
        if normalize:
            X = normalize_data(X)
        if tabulation == 'interp':
            X = interp_list_of_sequences(X)
        elif tabulation == 'obspad':
            X = pad_list_of_sequences(X)
        elif tabulation == 'zeropad':
            X = pad_list_of_sequences(X, pad_with=0.)
        X = np.asarray(X)
    
    len_examples = X.shape[1]
    num_features = X.shape[2]

    X = X.reshape([num_X, -1])
    if Y is not None:
        Y = Y.reshape([num_Y, -1])    
        return X, Y, len_examples, num_features
    else:
        return X, len_examples, num_features


### Helpers for kernel computations ###

def normalize_kernel_matrices(K_XX, K_XY=None, K_YY=None, jitter=1e-12):
    
    assert (K_XY is None) == (K_YY is None)
    
    K_XX += jitter
    K_X = np.diag(K_XX)

    if K_XY is None:
        K_XX /= np.sqrt(K_X[:, None] * K_X[None, :])
        return K_XX
    else:
        K_XY += jitter
        K_YY += jitter
        
        K_Y = np.diag(K_YY)
    
        K_XY /= np.sqrt(K_X[:, None] * K_Y[None, :])
        K_XX /= np.sqrt(K_X[:, None] * K_X[None, :])
        K_YY /= np.sqrt(K_Y[:, None] * K_Y[None, :])
        return K_XX, K_XY, K_YY
    

def compute_kernel_in_batches(kernel_fn, X, Y, batch_size=None):
    if batch_size is not None:
        num_X, num_Y = X.shape[0], Y.shape[0]
        K = np.zeros((num_X, num_Y))
        for i in range(int(np.ceil(float(num_X)/batch_size))):
            lower_X, upper_X = i*batch_size, min((i+1)*batch_size, num_X)
            batch_X = X[lower_X:upper_X]
            for j in range(int(np.ceil(float(num_Y)/batch_size))):
                lower_Y, upper_Y = j*batch_size, min((j+1)*batch_size, num_Y)
                batch_Y = Y[lower_Y:upper_Y]
                K[lower_X:upper_X, lower_Y:upper_Y] = kernel_fn(batch_X, batch_Y)
    else:
        K = kernel_fn(X, Y)
    return K

def compute_feature_in_batches(feature_fn, X, batch_size=100):
    if batch_size is not None:
        num_X = X.shape[0]
        P = []
        for i in range(int(np.ceil(float(num_X)/batch_size))):
            lower_X, upper_X = i*batch_size, min((i+1)*batch_size, num_X)
            batch_X = X[lower_X:upper_X]
            P.append(feature_fn(batch_X))
        P = np.concatenate(P, axis=0)
    else:
        P = feature_fn(X)
    return P

def compute_vector_kernel_matrices(kernel, X, Y=None, preprocess=True, tabulation=None, normalize_data=False, normalize_kernel=False, batch_size=None, **kwargs):
    
    if not preprocess and normalize_data:
        raise ValueError('Error: preprocessing must be enabled to normalize the data.')

    if preprocess:
        if Y is not None:
            X, Y, _, _ = preprocess_data(X, Y, tabulation=tabulation, normalize=normalize_data)
        else:
            X, _, _ = preprocess_data(X, tabulation=tabulation, normalize=normalize_data)
    
    if isinstance(kernel, type(lambda x:x)):
        kernel_fn = lambda _X, _Y: kernel(_X, _Y, **kwargs)
    elif issubclass(kernel, GPFlowKernel):
        k = kernel(**kwargs)
        kernel_fn = lambda X, Y: k.K(X, Y).numpy()
    elif issubclass(kernel, SkGPKernel):
        k = kernel(**kwargs)
        kernel_fn = lambda X, Y: k(X, Y)
    else:
        raise RuntimeError('Kernel from unknown class {}'.format(kernel))
    
    K_XX = compute_kernel_in_batches(kernel_fn, X, X, batch_size=batch_size)
    if Y is not None:
        K_XY = compute_kernel_in_batches(kernel_fn, X, Y, batch_size=batch_size)
        K_YY = compute_kernel_in_batches(kernel_fn, Y, Y, batch_size=batch_size)
        if normalize_kernel:
            K_XX, K_XY, K_YY = normalize_kernel_matrices(K_XX, K_XY, K_YY)
        return K_XX, K_XY, K_YY
    else:
        if normalize_kernel:
            K_XX = normalize_kernel_matrices(K_XX)
        return K_XX

def compute_gpsig_kernel_matrices(kernel, X, Y=None, low_rank=False, add_time=False, normalize_data=False, normalize_levels=False, normalize_kernel=False, batch_size=None, **kwargs):
    
    if X[0].ndim < 2: 
        raise RuntimeError('GPSig: Data must be passed in non-flattened format')
    
    if isinstance(X, list) or len(np.unique([x.shape[0] for x in X])) > 1:
        if Y is not None:
            X, Y, len_examples, num_features = preprocess_data(X, Y, normalize=normalize_data, tabulation='obspad')
        else:
            X, len_examples, num_features = preprocess_data(X, normalize=normalize_data, tabulation='obspad')
    else:
        len_examples, num_features = X[0].shape


    if add_time:
        X = add_time_to_table(X, num_features=num_features)
        if Y is not None:
            Y = add_time_to_table(Y, num_features=num_features)
        num_features += 1
    
    if not low_rank:
        k = kernel(num_features, len_examples, **kwargs)
        kernel_fn = lambda X, Y: k.K(X, Y).numpy()
        K_XX = compute_kernel_in_batches(kernel_fn, X, X, batch_size=batch_size)
        if Y is not None:
            K_XY = compute_kernel_in_batches(kernel_fn, X, Y, batch_size=batch_size)
            K_YY = compute_kernel_in_batches(kernel_fn, Y, Y, batch_size=batch_size)
    else:
        k = kernel(num_features, len_examples, low_rank=True, normalization=normalize_levels, **kwargs)
        if Y is not None:
            k.fit_low_rank_params(np.concatenate((X, Y), axis=0))
        else:
            k.fit_low_rank_params(X)
        feature_fn = lambda X: k.feat(X).numpy()
        P_X = compute_feature_in_batches(feature_fn, X, batch_size=batch_size)
        K_XX = P_X @ P_X.T
        if Y is not None:
            P_Y = compute_feature_in_batches(feature_fn, Y, batch_size=batch_size)
            K_XY = P_X @ P_Y.T
            K_YY = P_Y @ P_Y.T
    
    if Y is not None:
        if normalize_kernel:
            K_XX, K_XY, K_YY = normalize_kernel_matrices(K_XX, K_XY, K_YY)
        return K_XX, K_XY, K_YY
    else:
        if normalize_kernel:
            K_XX = normalize_kernel_matrices(K_XX)
        return K_XX 

def compute_kernel_matrices(kernel, X, Y=None, batch_size=None, **kwargs):
    if not isinstance(kernel, type(lambda x:x)) and issubclass(kernel, SignatureKernel):
        return compute_gpsig_kernel_matrices(kernel, X, Y, batch_size=batch_size, **kwargs)
    else:
        return compute_vector_kernel_matrices(kernel, X, Y, batch_size=batch_size, **kwargs)


### MMD computations ###

def quadratic_time_mmd(K_XX, K_XY, K_YY):
    
    n = np.shape(K_XX)[0]
    m = np.shape(K_YY)[0]
    
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    mmd = np.sum(K_XX) / (n*(n-1))  + np.sum(K_YY) / (m*(m-1))  - 2*np.sum(K_XY)/(n*m)
        
    return mmd

def _mmd_max(compute_kernel_matrices, X, Y, params_grid, verbose=False, name=None):

    times = []
    
    d_opt = -1.0
    p_opt = 0
    if verbose:
        if name is not None:
            print(f'Kernel: {name}')
        print('------------------')
        params_grid = tqdm(enumerate(params_grid), total=len(params_grid))
    else:
        params_grid = enumerate(params_grid)
        
    for i, p in params_grid: 
        st = time.time()
        
        K_XX, K_XY, K_YY = compute_kernel_matrices(X, Y, **p)
        
        d = quadratic_time_mmd(K_XX, K_XY, K_YY)
        
        if d > d_opt:
            d_opt = d
            p_opt = p
        times.append(time.time() - st)
    
    t_sum = np.sum(times)
    t_avg = np.mean(times)
    
    if verbose:
        print(f'Time elapsed: {t_sum:0.2f}')
        print(f'Time per iteration: {t_avg:0.2f}')
        print(f'Number of combinations: {len(params_grid)}')
        print(f'Found parameters: {p_opt}')
        print(f'MMD: {d_opt:0.3e}')
        print('------------------')
        print()
    return d_opt


def mmd_max(kernel, X, Y, params_grid, batch_size=None, verbose=False, name=None):
    
    d = _mmd_max(partial(compute_kernel_matrices, kernel, batch_size=batch_size), X, Y, params_grid, verbose=verbose, name=name)
    return d

### MMD-Max Permutation test ###

def _dict2key(p):
    return tuple(sorted(p.items())) 

def _precompute_kernel_matrices(compute_kernel_matrices, Z, params_grid, kernel_matrices=None, verbose=False):

    if verbose:
        params_grid = tqdm(params_grid, total=len(params_grid))
     
    # precompute kernel matrices for all parameter combinations (that are not already available)
    kernel_matrices = kernel_matrices or {}
    for i, p in enumerate(params_grid):
        key = _dict2key(p) 
        if key not in kernel_matrices:
            kernel_matrices[key] = compute_kernel_matrices(Z, **p)

    return kernel_matrices

def precompute_kernel_matrices(kernel, Z, params_grid, batch_size=None, kernel_matrices=None, verbose=False):
    return _precompute_kernel_matrices(partial(compute_kernel_matrices, kernel, batch_size=batch_size), Z, params_grid, kernel_matrices=kernel_matrices, verbose=verbose)
    

def mmd_max_permutation_test(kernel, X, Y, params_grid, num_permutations, batch_size=None, kernel_matrices=None, verbose=False):
    
    num_X = len(X)
    num_Y = len(Y)
    num_Z = num_X + num_Y
    
    # assert X.ndim == Y.ndim
    # concatenate samples
    assert isinstance(X, list) == isinstance(Y, list)
    if isinstance(X, list):
        Z = X + Y
    else:
        Z = np.concatenate((X, Y), axis=0)

    # precompute kernel matrices for all parmeter sets
    kernel_matrices = precompute_kernel_matrices(kernel, Z, params_grid, batch_size=batch_size, kernel_matrices=kernel_matrices, verbose=verbose)
        
    statistics = np.zeros(num_permutations)
    permutations = range(num_permutations)
    if verbose:
        permutations = tqdm(permutations)
    for i in permutations:
        perm_inds = np.random.permutation(num_Z)
        d_opt = -1
        for j, p in enumerate(params_grid):
            key = _dict2key(p)
            K_XX = kernel_matrices[key][np.ix_(perm_inds[:num_X], perm_inds[:num_X])]
            K_XY = kernel_matrices[key][np.ix_(perm_inds[:num_X], perm_inds[num_X:])]
            K_YY = kernel_matrices[key][np.ix_(perm_inds[num_X:], perm_inds[num_X:])]
            d = quadratic_time_mmd(K_XX, K_XY, K_YY)
            if d > d_opt:
                d_opt = d
        statistics[i] = d_opt

    d_opt = -1
    for i, p in enumerate(params_grid):
        key = _dict2key(p)
        K_XX = kernel_matrices[key][:num_X, :num_X]
        K_XY = kernel_matrices[key][:num_X, num_X:]
        K_YY = kernel_matrices[key][num_X:, num_X:]
        d = quadratic_time_mmd(K_XX, K_XY, K_YY)
        if d > d_opt:
            d_opt = d
    return statistics, d_opt

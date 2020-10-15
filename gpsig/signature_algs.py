import numpy as np
import tensorflow as tf

from gpflow import config

from .low_rank_calculations import lr_hadamard_prod_rand

def signature_kern_first_order(M, num_levels, difference = True):
    """
    Compute the first-order signature kernel matrix
    # Input
    :M:                 (num_examples1, len_examples1, num_examples2, len_examples2) or (num_examples, len_examples, len_examples) tensors
    :num_levels:        number of signature levels to compute
    # Output
    :K:                 (num_examples1, num_examples2) or (num_examples) tensor
    """

    if M.shape.ndims == 4: 
        num_examples1, len_examples1, num_examples2, len_examples2 = tf.unstack(tf.shape(M)[-4:])
        K = [tf.ones((num_examples1, num_examples2), dtype=config.default_float())]
    else:
        num_examples, len_examples = tf.shape(M)[0], tf.shape(M)[1]
        K = [tf.ones((num_examples), dtype=config.default_float())]
    
    if difference:
        M = M[:, 1:, ..., 1:] + M[:, :-1, ..., :-1] - M[:, :-1, ..., 1:] - M[:, 1:, ..., :-1]
    
    K.append(tf.reduce_sum(M, axis=(1, -1)))
    
    R = M
    for i in range(2, num_levels+1):
        R = M * tf.cumsum(tf.cumsum(R, exclusive=True, axis=1), exclusive=True, axis=-1)
        K.append(tf.reduce_sum(R, axis=(1, -1)))

    return tf.stack(K, axis=0)

def signature_kern_higher_order(M, num_levels, order=2, difference = True):
    """
    Compute the higher-order signature kernel matrix
    # Input
    :M:                 (num_examples1, len_examples1, num_examples2, len_examples2) or (num_examples, len_examples, len_examples) tensors
    :num_levels:        number of signature levels to compute
    :order:             order of approximation to use in signature kernel
    # Output
    :K:                 (num_examples1, num_examples2) or (num_examples) tensor
    """

    if M.shape.ndims == 4:
        num_examples1, num_examples2 = tf.shape(M)[0], tf.shape(M)[2]
        K = [tf.ones((num_examples1, num_examples2), dtype=config.default_float())]
    else:
        num_examples = tf.shape(M)[0]
        K = [tf.ones((num_examples), dtype=config.default_float())]
    
    if difference:
        M = M[:, 1:, ..., 1:] + M[:, :-1, ..., :-1] - M[:, :-1, ..., 1:] - M[:, 1:, ..., :-1]
    
    K.append(tf.reduce_sum(M, axis=(1, -1)))
    
    R = np.asarray([[M]])
    for i in range(2, num_levels+1):
        d = min(i, order)
        R_next = np.empty((d, d), dtype=tf.Tensor)
        R_next[0, 0] = M * tf.cumsum(tf.cumsum(tf.add_n(R.flatten().tolist()), exclusive=True, axis=1), exclusive=True, axis=-1)
        for j in range(2, d+1):
            R_next[0, j-1] = 1 / tf.cast(j, config.default_float()) * M * tf.cumsum(tf.add_n(R[:, j-2].tolist()), exclusive=True, axis=1)
            R_next[j-1, 0] = 1 / tf.cast(j, config.default_float()) * M * tf.cumsum(tf.add_n(R[j-2, :].tolist()), exclusive=True, axis=-1)
            for k in range(2, d+1):
                R_next[j-1, k-1] = 1 / (tf.cast(j, config.default_float()) * tf.cast(k, config.default_float())) * M * R[j-2, k-2]

        K.append(tf.reduce_sum(tf.add_n(R_next.flatten().tolist()), axis=(1, -1)))
        R = R_next
    
    return tf.stack(K, axis=0)

def tensor_kern(M, num_levels):
    """
    Computing the square matrix of inner product of inducing tensors
    # Input
    :M:                 (num_levels*(num_levels+1)/2, num_tensors, num_tensors) tensor components vs tensor components kernel matrices or 
                        (num_levels*(num_levels+1)/2, num_tensors)
    :num_levels:        degree of truncation for the signatures
    # Output
    :K:                 (num_tensors, num_tensors) kernel matrix tensors
    """

    if M.shape.ndims == 3:
        num_tensors, num_tensors2 = tf.shape(M)[1], tf.shape(M)[2]
        K = [tf.ones((num_tensors, num_tensors2), dtype=config.default_float())]
    else:
        num_tensors = tf.shape(M)[1]
        K = [tf.ones((num_tensors), dtype=config.default_float())]
    
    k = 0
    for i in range(1, num_levels+1):
        R = M[k]
        k += 1
        for j in range(1, i):
            R = M[k] * R
            k += 1
        K.append(R)

    return tf.stack(K, axis=0)
        
def signature_kern_tens_vs_seq_first_order(M, num_levels, difference = True):
    """
    Compute tensor vs (first-order) signature inner products
    # Input
    :M:                 (num_levels*(num_levels+1)/2, num_tensors, num_examples, len_examples)
    :num_levels:        degree of truncation for the signatures
    # Output
    :K:                 (num_tensors, num_examples) inner product matrix
    """

    num_tensors, num_examples, len_examples = tf.unstack(tf.shape(M)[1:])

    if difference:
        M = M[..., 1:] - M[..., :-1] # difference along time series axis

    K = [tf.ones((num_tensors, num_examples), dtype=config.default_float())]
    
    k = 0
    for i in range(1, num_levels+1):
        R = M[k]
        k += 1
        for j in range(1, i):
            R = M[k] * tf.cumsum(R, exclusive=True, axis=2)
            k += 1
        K.append(tf.reduce_sum(R, axis=2))

    return tf.stack(K, axis=0)

def signature_kern_tens_vs_seq_higher_order(M, num_levels, order=2, difference = True):
    """
    Compute tensor vs (higher-order) signature inner products
    # Input
    :M:                 (num_levels*(num_levels+1)/2, num_tensors, num_examples, len_examples)
    :num_levels:        degree of truncation for the signatures
    # Output
    :K:                 (num_tensors, num_examples) inner product matrix
    """

    num_tensors, num_examples, len_examples = tf.unstack(tf.shape(M)[1:])

    if difference:
        M = M[..., 1:] - M[..., :-1] # difference along time series axis

    K = [tf.ones((num_tensors, num_examples), dtype=config.default_float())]
    
    k = 0
    for i in range(1, num_levels+1):
        R = np.asarray([M[k]])
        k += 1
        for j in range(1, i):
            d = min(j+1, order)
            R_next = np.empty((d), dtype=tf.Tensor)
            R_next[0] = M[k] * tf.cumsum(tf.add_n(R.tolist()), exclusive=True, axis=2)
            for l in range(1, d):
                R_next[l] = 1. / tf.cast(l+1, config.default_float()) * M[k] * R[l-1]
            R = R_next
            k += 1
        K.append(tf.reduce_sum(tf.add_n(R.tolist()), axis=2))
    
    return tf.stack(K, axis=0)

def signature_kern_first_order_lr_feature(U, num_levels, rank_bound, sparsity = 'sqrt', seeds = None, difference = True):
    """
    Compute feature map for (first-order) low-rank signatures from low-rank factor of big kernel matrix
    # Input
    :U:                 (num_examples, len_examples, rank_bound) low-rank feature representations for embedded sequences
    :num_levels:        degree of truncation for the signatures
    :rank_bound:        number of components used in the low-rank approximation
    :sparsity:          controls the sparsity of the randomized projection used for simplifying the low-rank factor at every iteration
                        possible values are: 'sqrt' - most accurate, but costly; 'log' - less accurate, but cheaper; 'lin' - sparsest, least accurate
    :seeds:             optional - (num_levels-1, 2) random seeds for random projections
    # Output
    :Phi:               (num_levels+1,) list of low-rank factors               
    """
     
    num_examples, len_examples, rank_bound = tf.unstack(tf.shape(U))
    Phi = [tf.ones((num_examples, 1), dtype=config.default_float())]

    if difference:
        U = U[:, 1:, :] - U[:, :-1, :]
        
    Phi.append(tf.reduce_sum(U, axis=1))

    P = U
    for i in range(2, num_levels+1): 
        P = tf.cumsum(P, axis=1, exclusive=True)
        if seeds is None:
            P = lr_hadamard_prod_rand(U, P, rank_bound, sparsity)
        else:
            P = lr_hadamard_prod_rand(U, P, rank_bound, sparsity, seeds[i-2])
        Phi.append(tf.reduce_sum(P, axis=1))
    return Phi

def tensor_kern_lr_feature(U, num_levels, rank_bound, sparsity = 'sqrt', seeds = None):
    """
    Compute the low-rank feature map for tensors
    # Input
    :U:                 (num_levels*(num_levels+1)/2, num_tensors, rank_bound) low-rank feature representations for inducing tensors
    :num_levels:        degree of truncation for the signatures
    :rank_bound:        number of components used in the low-rank approximation
    :sparsity:          controls the sparsity of the randomized projection used for simplifying the low-rank factor at every iteration
                        possible values are: 'sqrt' - most accurate, but costly; 'log' - less accurate, but cheaper; 'lin' - sparsest, least accurate
    :seeds:             optional - (num_levels-1, 2) random seeds for random projections
    # Output
    :Phi:               (num_levels+1,) list of low-rank factors for tensors               
    """

    num_tensors = tf.shape(U)[1]
    Phi = [tf.ones((num_tensors, 1), dtype=config.default_float())]

    k = 0
    for i in range(1, num_levels+1):
        R = U[k]
        k += 1
        for j in range(1, i):
            if seeds is None:
                R = lr_hadamard_prod_rand(U[k], R, rank_bound, sparsity)
            else:
                R = lr_hadamard_prod_rand(U[k], R, rank_bound, sparsity, seeds[j-1])
            k += 1
        Phi.append(R)
    return Phi
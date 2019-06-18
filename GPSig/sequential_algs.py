import numpy as np
import tensorflow as tf

from gpflow import settings

from . import low_rank_calculations

def sequentialize_kern_symm(M, variances, num_levels, normalize_levels = True, difference = True, return_level_norms = False):
    """
    Full-rank first order sequentializer algorithm for computing the sequential kernel of a symmetric square kernel matrix.
    Input
    :M:                 (num_streams, stream_length, num_streams, stream_length) array of a time samples vs time samples kernel matrices
    :variances:         (num_levels + 1,) vector (tensor) of coefficients for each level in the inner product of truncated signatures
    :num_levels:        degree of truncation for the signatures
    :normalize_levels:  boolean variable indicating whether to normalize the signatures levels using the norm
    :return_level_norms:  boolean indicating whether to return as a second output the unnormalized diagonals of each level independently
                        (only taken into account if normalize_levels==True)
    Output
    :K:                 (num_streams, num_streams) sequentialized kernel matrix or (num_streams) diagonal entries of the sequentialized kernel
    :K_i_norms:       (num_levels, num_streams)
    """

    # assert M.shape[0] == M.shape[2]
    # assert M.shape[1] == M.shape[3]

    num_streams = tf.shape(M)[0]
    
    if difference:
        M = M[:, 1:, :, 1:] + M[:, :-1, :, :-1] - M[:, :-1, :, 1:] - M[:, 1:, :, :-1]
    
    variances = variances * tf.ones((num_levels + 1), dtype=settings.float_type)
    K = variances[0] * tf.ones((num_streams, num_streams), dtype=settings.float_type)
    R = M
    K_i = tf.reduce_sum(tf.reduce_sum(R, axis=-3), axis=-1) + settings.numerics.jitter_level * tf.eye(num_streams, dtype=settings.float_type)
    if normalize_levels:
        norm = tf.sqrt(tf.diag_part(K_i))
        # norm = tf.maximum(tf.diag_part(K_i), settings.numerics.jitter_level)
        K_i /= norm[:, None] * norm[None, :]
        if return_level_norms:
            K_i_norms = norm[None, :]
    K += variances[1] * K_i
    for i in range(2, num_levels+1):
        R = tf.cumsum(tf.cumsum(R, exclusive=True, axis=1), exclusive=True, axis=-1)
        R = M * R
        K_i = tf.reduce_sum(tf.reduce_sum(R, axis=1), axis=-1) + settings.numerics.jitter_level * tf.eye(num_streams, dtype=settings.float_type)
        if normalize_levels:
            norm = tf.sqrt(tf.diag_part(K_i))
            # norm = tf.maximum(tf.sqrt(tf.diag_part(K_i)), settings.numerics.jitter_level)
            K_i /= norm[..., :, None] * norm[..., None, :]
            if return_level_norms:
                K_i_norms = tf.concat((K_i_norms, norm[None, :]), axis=0)
        K += variances[i] * K_i
    if return_level_norms:
        return K, K_i_norms
    else:
        return K

def sequentialize_kern_diag(M, variances, num_levels, normalize_levels = True, difference = True, return_level_norms = False):
    """
    Full-rank first order sequentializer algorithm for computing the unnormalized level diagonals of a first-order sequential kernel matrix.
    Input
    :M:                     (num_streams, stream_length, stream_length) tensor of time samples vs time samples kernel matrices
    :variances:             (num_levels + 1,) tensor of coefficients for each level
    :num_levels:            degree of truncation for the signatures
    :normalize_levels:      boolean indicating whether to return normalized_diagonals (e.g. Sum_i sigma_i)
    :return_level_norms:    whether to return the sqrt of each diagonal component of signature levels
    Output 
    :K:                 (num_streams) tensor of diagonals
    :K_i_norms:         (num_levels, num_streams) tensor of level norms (optional)
    """

    num_streams = tf.shape(M)[0]
    variances = variances * tf.ones((num_levels + 1), dtype=settings.float_type)
    if normalize_levels and not return_level_norms:
        return tf.reduce_sum(variances) * tf.ones((num_streams), dtype=settings.float_type)
    
    if difference:
        M = M[:, 1:, 1:] + M[:, :-1, :-1] - M[:, :-1, 1:] - M[:, 1:, :-1]
    
    R = M
    K_i_diags = tf.reduce_sum(tf.reduce_sum(R, axis=1), axis=-1)[None, :]
    for i in range(2, num_levels+1):
        R = tf.cumsum(tf.cumsum(R, exclusive=True, axis=1), exclusive=True, axis=-1)
        R = M * R
        K_i_diags = tf.concat((K_i_diags, tf.reduce_sum(tf.reduce_sum(R, axis=-1), axis=-1)[None, :]), axis=0)
    
    K_i_diags += settings.numerics.jitter_level
    if normalize_levels:
        K_i_norms = tf.sqrt(K_i_diags)
        # K_i_norms = tf.maximum(tf.sqrt(K_i_diags), settings.numerics.jitter_level)
        return tf.reduce_sum(variances) * tf.ones((num_streams), dtype=settings.float_type), K_i_norms
    else:
        K_diag = variances[0] + tf.reduce_sum(variances[1:, None] * K_i_diags, axis=0)
        if return_level_norms:
            K_i_norms = tf.sqrt(K_i_diags)
            # K_i_norms = tf.maximum(tf.sqrt(K_i_diags), settings.numerics.jitter_level)
            return K_diag, K_i_norms
        else:
            return K_diag
    
def sequentialize_kern_rect(M, variances, num_levels, normalize_levels = True, difference = True, K_i_norms1 = None, K_i_norms2 = None):
    """
    Full-rank first order sequentializer algorithm for computing the sequential kernel of a rectangular kernel matrix.
    Input
    :M:                 (num_streams1, stream_length, num_streams2, stream_length2) tensor of time samples vs time samples kernel matrices
    :variances:         (num_levels + 1,) tensor of coefficients for the signature levels
    :num_levels:        integer representing the cut-off degree for the signatures
    :normalize_levels:  boolean variable indicating whether to normalize the signatures levels using the norm
    :K_i_norms1:        (num_levels, num_streams1) tensor of level norms, only taken into account if normalize_levels==True
    :K_i_norms2:        (num_levels, num_streams2) tensor of level norms, only taken into account if normalize_levels==True
    Output
    :K:                 (num_streams1, num_streams2) sequentialized kernel matrix
    """
    
    num_streams1, num_streams2 = tf.shape(M)[0], tf.shape(M)[2]
    if normalize_levels == True:
        assert K_i_norms1 is not None # and tf.shape(K_i_norms1)[0] == num_levels # and tf.shape(K_i_norms1)[1] == num_streams1
        assert K_i_norms2 is not None # and tf.shape(K_i_norms2)[0] == num_levels # and tf.shape(K_i_norms2)[1] == num_streams2

    if difference:
        M = M[:, 1:, :, 1:] + M[:, :-1, :, :-1] - M[:, :-1, :, 1:] - M[:, 1:, :, :-1]

    variances = variances * tf.ones((num_levels + 1), dtype=settings.float_type)
    K = variances[0] * tf.ones((num_streams1, num_streams2), dtype=settings.float_type)
    R = M
    for i in range(1, num_levels):
        K_i = tf.reduce_sum(tf.reduce_sum(R, axis=1), axis=-1)
        if normalize_levels:
            K_i /= K_i_norms1[i-1, :, None] * K_i_norms2[i-1, None, :]
        K += variances[i] * K_i
        R = tf.cumsum(tf.cumsum(R, exclusive=True, axis=1), exclusive=True, axis=-1)
        R = M * R
    K_i = tf.reduce_sum(tf.reduce_sum(R, axis=1), axis=-1)
    if normalize_levels:
        K_i /= K_i_norms1[-1, :, None] * K_i_norms2[-1, None, :]
    K += variances[i] * K_i
    return K

        
def sequentialize_kern_inter(M, variances, num_levels, normalize_levels = True, difference = True, K_i_norms = None):
    """
    Full-rank first order sequentializer algorithm for computing the inter-domain cross-covariances.
    Input
    :M:                 (num_levels*(num_levels+1)/2, num_inducing, num_streams, stream_length) tensor of inducing-inputs vs time-samples kernel matrices
    :variances:         (num_levels + 1,) tensor of coefficients for each level in the inner product of truncated signatures
    :num_levels:        degree of truncation for the signatures
    :normalize_levels:  boolean variable indicating whether to normalize the signatures levels using the norm
    :K_i_norms:         None or a tensor of (num_levels, num_streams) if normalize_levels==True
    Output
    :Kzx:               (num_inducing, num_streams) inter-domain cross-covariance sequentialized kernel matrix
    """

    if normalize_levels:
        assert K_i_norms is not None
    
    num_inducing, num_streams = tf.shape(M)[1], tf.shape(M)[2]

    if difference:
        M = M[..., 1:] - M[..., :-1] # difference along time series axis
    variances = variances * tf.ones((num_levels + 1), dtype=settings.float_type)
    Kzx = variances[0] * tf.ones((num_inducing, num_streams), dtype=settings.float_type)
    k = 0
    for i in range(1, num_levels+1):
        R = M[k, ...]
        k += 1
        for j in range(1, i):
            R = tf.cumsum(R, exclusive=True, axis=-1)
            R = M[k, ...] * R
            k += 1
        Kzx_i = tf.reduce_sum(R, axis=-1)
        if normalize_levels:
            Kzx_i /= K_i_norms[i-1, None, :]
        Kzx += variances[i] * Kzx_i
    return Kzx

        
def sequentialize_kern_inducing(M, variances, num_levels):
    """
    Full-rank first order sequentializer algorithm for computing the inducing-point covariances.
    Input
    :M:                 (num_levels*(num_levels+1)/2, num_inducing, num_inducing) tensor of inducing-inputs vs inducing-inputs kernel matrices
    :variances:         (num_levels + 1,) tensor of coefficients for the signature levels
    :num_levels:        degree of truncation for the signatures
    Output
    :Kuu:                 (num_inducing, num_inducing) sequentialized kernel matrix or (num_streams) diagonal entries of the sequentialized kernel
    """

    num_inducing = tf.shape(M)[1]
    variances = variances * tf.ones((num_levels + 1), dtype=settings.float_type)
    Kuu = variances[0] * tf.ones((num_inducing, num_inducing), dtype=settings.float_type)
    k = 0
    for i in range(1, num_levels+1):
        R = M[k]
        k += 1
        for j in range(1, i):
            R = M[k] * R
            k += 1
        Kuu += variances[i] * R
    return Kuu

def sequentialize_kern_low_rank(U, variances, num_levels, rank_bound, sparsity = 'sqrt', normalize_levels = True, difference = True, seeds = None):
    """
    Low-rank first order sequentializer algorithm for computing the sequential kernel.
    Input
    :U:                 (num_streams, stream_length, num_components) array of kernel matrix low-rank factors
    :variances:         (num_levels + 1,) vector (tensor) of coefficients for each level in the inner product of truncated signatures
    :num_levels:        degree of truncation for the signatures
    :rank_bound:        maximum size of the low-rank factor
    :sparsity:          controls the sparsity of the randomized projection used for simplifying the low-rank factor at every iteration
                        possible values are: 'sqrt' - most accurate, but costly; 'log' - less accurate, but cheaper; 'subsample' - sparsest, least accurate
    :normalize_levels:  boolean variable indicating whether to normalize the signatures levels independently using the norm
    Output 
    :K:                 (num_streams, num_streams) approximate sequentialized kernel matrix
    """
    
    if difference:
        U = U[..., 1:, :] - U[..., :-1, :] # difference along axis=-2
    
    P = U 
    # break down shape of U
    batch_shape = tf.shape(U)[:-2]
    # shape of output K
    K_shape = tf.concat([batch_shape, [batch_shape[-1]]], axis=0)
    # init variances
    variances = variances * tf.ones((num_levels + 1), dtype=settings.float_type)
    # initialise K
    K = variances[0] * tf.ones(K_shape, dtype=settings.float_type)
    for i in range(1, num_levels):
        P_reduce = tf.reduce_sum(P, axis=-2)
        K_i =  tf.matmul(P_reduce, P_reduce, transpose_b=True)
        if normalize_levels:
            # norm = tf.sqrt(tf.diag_part(K_i)) + settings.numerics.jitter_level
            norm = tf.maximum(tf.sqrt(tf.diag_part(K_i)), settings.numerics.jitter_level)
            K_i /= norm[..., :, None] * norm[..., None, :]
        K += variances[i] * K_i
        P = tf.cumsum(P, axis=-2, exclusive=True)
        if sparsity == 'subsample':
            if seeds is None:
                P = low_rank_calculations.lr_hadamard_prod_subsample(U, P, rank_bound)
            else:
                P = low_rank_calculations.lr_hadamard_prod_subsample(U, P, rank_bound, seeds[i-1])
        elif sparsity == 'subsample_gauss':
            if seeds is None:
                P = low_rank_calculations.lr_hadamard_prod_subsample_gauss(U, P, rank_bound)
            else:
                P = low_rank_calculations.lr_hadamard_prod_subsample_gauss(U, P, rank_bound, seeds[i-1])
        else:
            if seeds is None:
                P = low_rank_calculations.lr_hadamard_prod_sparse(U, P, rank_bound, sparsity)
            else:
                P = low_rank_calculations.lr_hadamard_prod_sparse(U, P, rank_bound, sparsity, seeds[i-1])

    P_reduce = tf.reduce_sum(P, axis=-2)
    K_i =  tf.matmul(P_reduce, P_reduce, transpose_b=True)
    if normalize_levels:
        # norm = tf.sqrt(tf.diag_part(K_i)) + settings.numerics.jitter_level
        norm = tf.maximum(tf.sqrt(tf.diag_part(K_i)), settings.numerics.jitter_level)
        K_i /= norm[..., :, None] * norm[..., None, :]
    K += variances[-1] * K_i
    return K

def sequentialize_kern_low_rank_rect(U, V, variances, num_levels, rank_bound, sparsity = 'sqrt', normalize_levels = True, difference = True, seeds = None):
    """
    Low-rank first order sequentializer algorithm for computing a rectangular signature kernel matrix.
    Input
    :U:                 (num_streams, stream_length, num_components) array of kernel matrix low-rank factors
    :V:                 (num_streams2, stream_length2, num_components) array of kernel matrix low-rank factors
    :variances:         (num_levels + 1,) vector (tensor) of coefficients for each level in the inner product of truncated signatures
    :num_levels:        degree of truncation for the signatures
    :rank_bound:        maximum size of the low-rank factor
    :sparsity:          controls the sparsity of the randomized projection used for simplifying the low-rank factor at every iteration
                        possible values are: 'sqrt' - most accurate, but costly; 'log' - less accurate, but cheaper; 'subsample' - sparsest, least accurate
    :normalize_levels:  boolean variable indicating whether to normalize the signatures levels independently using the norm
    Output 
    :K:                 (num_streams, num_streams2) approximate sequentialized kernel matrix
    """
    if difference:
        U = U[..., 1:, :] - U[..., :-1, :]
        V = V[..., 1:, :] - V[..., :-1, :]
    
    P = U
    Q = V
    # break down shape of U
    batch1_shape = tf.shape(U)[:-2]
    # shape of output K
    K_shape = tf.concat((batch_shape, [tf.shape(V)[-2]]), axis=0)
    # init variances
    variances = variances * tf.ones((num_levels + 1), dtype=settings.float_type)
    # initialise K
    K = variances[0] * tf.ones(K_shape, dtype=settings.float_type)
    for i in range(1, num_levels):
        P_reduce = tf.reduce_sum(P, axis=-2)
        Q_reduce = tf.reduce_sum(Q, axis=-2)
        K_i =  tf.matmul(P_reduce, Q_reduce, transpose_b=True)
        if normalize_levels:
            # norm1 = tf.norm(P_reduce, axis=-1) + settings.numerics.jitter_level
            norm1 = tf.maximum(tf.norm(P_reduce, axis=-1), settings.numerics.jitter_level)
            # norm2 = tf.norm(Q_reduce, axis=-1) + settings.numerics.jitter_level
            norm2 = tf.maximum(tf.norm(Q_reduce, axis=-1), settings.numerics.jitter_level)
            K_i /= norm1[..., :, None] * norm2[..., None, :]
        K += variances[i] * K_i

        P = tf.cumsum(P, axis=-2, exclusive=True)
        Q = tf.cumsum(Q, axis=-2, exclusive=True)
        if seeds is None:
            seed = tf.random_uniform((2), minval=0, maxval=np.iinfo(settings.int_type).max, dtype=settings.int_type)
        else:
            seed = seeds[i-1]
        if sparsity == 'subsample':
            P = low_rank_calculations.lr_hadamard_prod_subsample(U, P, rank_bound, seed=seed)
            Q = low_rank_calculations.lr_hadamard_prod_subsample(V, Q, rank_bound, seed=seed)
        elif sparsity == 'subsample_gauss':
            P = low_rank_calculations.lr_hadamard_prod_subsample_gauss(U, P, rank_bound, seed=seed)
            Q = low_rank_calculations.lr_hadamard_prod_subsample_gauss(V, Q, rank_bound, seed=seed)
        else:
            P = low_rank_calculations.lr_hadamard_prod_sparse(U, P, rank_bound, sparsity, seed=seed)
            Q = low_rank_calculations.lr_hadamard_prod_sparse(V, Q, rank_bound, sparsity, seed=seed)
    
    P_reduce = tf.reduce_sum(P, axis=-2)
    Q_reduce = tf.reduce_sum(Q, axis=-2)
    K_i =  tf.matmul(P_reduce, Q_reduce, transpose_b=True)
    if normalize_levels:
        # norm1 = tf.norm(P_reduce, axis=-1) + settings.numerics.jitter_level
        norm1 = tf.maximum(tf.norm(P_reduce, axis=-1), settings.numerics.jitter_level)
        # norm2 = tf.norm(Q_reduce, axis=-1) + settings.numerics.jitter_level
        norm2 = tf.maximum(tf.norm(Q_reduce, axis=-1), settings.numerics.jitter_level)
        K_i /= norm1[..., :, None] * norm2[..., None, :]
    K += variances[-1] * K_i
    return K


def sequentialize_kern_low_rank_feature(U, variances, num_levels, rank_bound, sparsity = 'sqrt', normalize_levels = True, difference = True, seeds = None):
    """
    Low-rank first order sequentializer algorithm for computing the sequential kernel. Output is returned as a feature map.
    Input
    :U:                 (num_streams, stream_length, num_components) array of kernel matrix low-rank factors 
                        or (stream_length, num_components) low-rank factor
    :variances:         (num_levels + 1,) vector (tensor) of coefficients for each level in the inner product of truncated signatures
    :num_levels:        degree of truncation for the signatures
    :rank_bound:        number of components used in the low-rank approximation
    :sparsity:          controls the sparsity of the randomized projection used for simplifying the low-rank factor at every iteration
                        possible values are: 'sqrt' - most accurate, but costly; 'log' - less accurate, but cheaper; 'subsample' - sparsest, least accurate
    :normalize_levels:  boolean variable indicating whether to normalize the signatures levels independently using the norm
    :seeds:             optional - (num_levels, 2) random seeds for random projections
    Output
    :Phi:               (num_streams, num_levels * rank_bound + 1) feature map or (num_levels * rank_bound + 1) feature vector
    """
    
    if difference:
        U = U[..., 1:, :] - U[..., :-1, :] # difference along axis=-2
    
    P = U 
    
    batch_shape = tf.shape(U)[:-2]
    alpha = tf.sqrt(variances) * tf.ones((num_levels + 1), dtype=settings.float_type)
    Phi = alpha[0] * tf.ones(tf.concat((batch_shape, [1]), axis = 0), dtype=settings.float_type)

    for i in range(1, num_levels):
        Phi_i = tf.reduce_sum(P, axis=-2)
        if normalize_levels:
            # Phi_i /= (tf.norm(Phi_i, axis=-1) + settings.numerics.jitter_level)[..., None]
            Phi_i /= (tf.maximum(tf.norm(Phi_i, axis=-1), settings.numerics.jitter_level))[..., None]
        Phi = tf.concat((Phi, alpha[i] * Phi_i), axis=-1) 
        P = tf.cumsum(P, axis=-2, exclusive=True)
        if sparsity == 'subsample':
            if seeds is None:
                P = low_rank_calculations.lr_hadamard_prod_subsample(U, P, rank_bound)
            else:
                P = low_rank_calculations.lr_hadamard_prod_subsample(U, P, rank_bound, seeds[i-1])
        elif sparsity == 'subsample_gauss':
            if seeds is None:
                P = low_rank_calculations.lr_hadamard_prod_subsample_gauss(U, P, rank_bound)
            else:
                P = low_rank_calculations.lr_hadamard_prod_subsample_gauss(U, P, rank_bound, seeds[i-1])
        else:
            if seeds is None:
                P = low_rank_calculations.lr_hadamard_prod_sparse(U, P, rank_bound, sparsity)
            else:
                P = low_rank_calculations.lr_hadamard_prod_sparse(U, P, rank_bound, sparsity, seeds[i-1])
    Phi_i = tf.reduce_sum(P, axis=-2)
    if normalize_levels:
        # Phi_i /= (tf.norm(Phi_i, axis=-1) + settings.numerics.jitter_level)[..., None]
        Phi_i /= (tf.maximum(tf.norm(Phi_i, axis=-1), settings.numerics.jitter_level))[..., None]
    Phi = tf.concat((Phi, alpha[-1] * Phi_i), axis=-1)
    return Phi

def sequentialize_kern_inducing_low_rank_feature(U, variances, num_levels, rank_bound, sparsity = 'sqrt', seeds = None):
    """
    Low-rank first order algorithm for computing a signature feature map for the inter-domain inducing point.
    Input
    :U:                 (num_levels*(num_levels+1)/2, num_inducing, num_components) array of inducing-input kernel matrix low-rank factors
    :variances:         (num_levels + 1,) tensor of coefficients for the signature moments
    :num_levels:        degree of truncation for signatures
    :rank_bound:        maximum size of the low-rank factor
    :sparsity:          controls the sparsity of the randomized projection used for simplifying the low-rank factor at every iteration
                        possible values are: 'sqrt' - most accurate, but costly; 'log' - less accurate, but cheaper; 'subsample' - sparsest, least accurate
    Output
    :Phi_Z:             (num_levels*(num_levels+1)/2, num_inducing, num_levels*rank_bound+1) array of inducing-input factors
    """

    num_inducing = tf.shape(U)[1]
    alpha = tf.sqrt(variances) * tf.ones((num_levels + 1), dtype=settings.float_type)
    
    Phi_Z = alpha[0] * tf.ones((num_inducing, 1), dtype=settings.float_type)
    k = 0
    for i in range(1, num_levels+1):
        P = U[k]
        k += 1
        for j in range(1, i):
            if sparsity == 'subsample':
                if seeds is None:
                    P = low_rank_calculations.lr_hadamard_prod_subsample(P, U[k], rank_bound)
                else:
                    P = low_rank_calculations.lr_hadamard_prod_subsample(P, U[k], rank_bound, seeds[j-1])
            elif sparsity == 'subsample_gauss':
                if seeds is None:
                    P = low_rank_calculations.lr_hadamard_prod_subsample_gauss(P, U[k], rank_bound)
                else:
                    P = low_rank_calculations.lr_hadamard_prod_subsample_gauss(P, U[k], rank_bound, seeds[j-1])
            else:
                if seeds is None:
                    P = low_rank_calculations.lr_hadamard_prod_sparse(P, U[k], rank_bound, sparsity)
                else:
                    P = low_rank_calculations.lr_hadamard_prod_sparse(P, U[k], rank_bound, sparsity, seeds[j-1])
            k += 1
        Phi_Z = tf.concat((Phi_Z, alpha[i] * P), axis=-1)

    return Phi_Z
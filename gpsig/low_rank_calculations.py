import tensorflow as tf

from gpflow import settings
from gpflow.conditionals import base_conditional
import numpy as np
from tensorflow.contrib import stateless
import tensorflow as tf
from gpflow import settings
from gpflow.kullback_leiblers import gauss_kl


def _draw_indices(n, l, need_inv = False):
    """
    Draws l indices from 0 to n-1 without replacement.
    Returns of a list of drawn and not drawn indices, and the inverse permutation 
    """
    idx = tf.random_shuffle(tf.range(n))
    idx_sampled, idx_not_sampled = tf.split(idx, [l, n-l])
    if need_inv:
        inv_map = tf.reverse(tf.nn.top_k(idx, k = n, sorted = True)[1], axis = [0])
        return idx_sampled, idx_not_sampled, inv_map
    else:
        return idx_sampled, idx_not_sampled


def Nystrom_map(X, kern, nys_samples = None, num_components = None):
    """
    Computes the Nystrom features with uniform sampling given a kernel and num_components
    See e.g. https://dl.acm.org/citation.cfm?id=2343678
    -------------------------------------------------------------------
    # Input
    :X:                 (num_samples, num_dims) tensor of data point observations with size 
    :kern:              function handle to a kernel function that takes two matrices as input e.g. X1 (num_samples1, num_dims)
                        and X2 (num_samples2, num_dim), and computes the matrix k(X1, X2) matrix of size (num_samples1, num_samples2)
    :nys_samples:       if given, these samples are used in the Nystrom approximation, has priority over num_components
    :num_components:    number of components to take, i.e. the rank of the low-rank kernel matrix
    # Output
    :X_nys:             tensor of Nystrom features of shape (num_samples, num_components)
    """
    
    num_samples = tf.shape(X)[0]
    
    if nys_samples is None and num_components is None:
        raise ValueError('One of num_components or nys_samples should be given')

    if nys_samples is None:
        idx, _ = _draw_indices(num_samples, num_components)
        nys_samples = tf.gather(X, idx, axis = - 2)

    num_components = tf.shape(nys_samples)[0]
    W = kern(nys_samples, nys_samples)
    W += tf.diag(settings.numerics.jitter_level * tf.random_uniform([num_components], dtype=settings.float_type))
    # to get around some undeterminedness of the gradient in special cases
    
    S, U = tf.self_adjoint_eig(W)
    S += settings.jitter * tf.ones((num_components), dtype=settings.float_type)
    D = tf.sqrt(S)

    Kxy = kern(X, nys_samples)
    X_nys = tf.matmul(Kxy, U) / D[None, :]
    return X_nys


def lr_hadamard_prod(A, B):
    """
    Computes the low-rank equivalent of the Hadamard product between matrices, i.e. the outer product of feature-representations.
    # Input
    :A: An [..., k1] tensor
    :B: An [..., k2] tensor
    # Output
    :C: An [..., k1*k2] tensor
    """
    C = tf.matmul(tf.expand_dims(A, axis=-1), tf.expand_dims(B, axis=-2))
    return tf.reshape(C, tf.concat((tf.shape(C)[:-2], [tf.reduce_prod(tf.shape(C)[-2:], axis=0)]), axis=0))

def lr_hadamard_prod_rand(A, B, rank_bound, sparsity='sqrt', seeds=None):
    """
    Computes a randomized low-rank Hadamard product    
    # Input
    :A:         An [..., k1] tensor
    :B:         An [..., k2] tensor
    :sparsity:  Order of sparsity in random projection matrix
    Output
    :C:         An [..., rank_bound] tensor
    """
    if sparsity == 'lin':
        C = lr_hadamard_prod_subsample(A, B, rank_bound, seeds)
    else:
        C = lr_hadamard_prod_sparse(A, B, rank_bound, sparsity, seeds)
    return C

def _draw_n_rademacher_samples(n, seed = None):
    """
    Draws n rademacher samples.
    """
    if seed is None:
        return tf.where(tf.random_uniform([n], dtype=settings.float_type) <= 0.5,
                tf.ones([n], dtype=settings.float_type), -1.*tf.ones([n], dtype=settings.float_type))
    else:
        return tf.where(stateless.stateless_random_uniform([n], dtype=settings.float_type, seed = seed) <= 0.5,
                tf.ones([n], dtype=settings.float_type), -1.*tf.ones([n], dtype=settings.float_type))


def lr_hadamard_prod_subsample(A, B, num_components, seed = None):
    """
    Low-rank Hadamard product with subsampling.
    # Input
    :A: An [..., k1] tensor
    :B: An [..., k2] tensor
    # Output
    :return C: A [..., num_components] tensor
    """
    batch_shape = tf.shape(A)[:-1]
    k1 = tf.shape(A)[-1]
    k2 = tf.shape(B)[-1]
    idx1 = tf.reshape(tf.range(k1, dtype=settings.int_type), [1, -1, 1])
    idx2 = tf.reshape(tf.range(k2, dtype=settings.int_type), [-1, 1, 1])
    
    combinations = tf.concat([idx1 + tf.zeros_like(idx2), tf.zeros_like(idx1) + idx2], axis=2)
    combinations = tf.random_shuffle(tf.reshape(combinations, [-1, 2]))
    
    select = combinations[:num_components]
    A = tf.gather(A, select[:,0], axis=-1)
    B = tf.gather(B, select[:,1], axis=-1)
    C = tf.reshape(A * B, [-1, num_components])
    D = tf.expand_dims(_draw_n_rademacher_samples(num_components, seed = seed), axis=0)    
    return tf.reshape(C * D, tf.concat((batch_shape, [num_components]), axis=0))


def _draw_n_gaussian_samples(n, seed = None):
    """
    Draws n gaussian samples.
    """
    if seed is None:
        return tf.random_normal([n], dtype=settings.float_type)
    else:
        return stateless.stateless_random_normal([n], dtype=settings.float_type, seed = seed)

def _draw_n_sparse_gaussian_samples(n, s, seed = None):
    """
    Draws n sparse gaussian samples, that is with P(X = N(0,1)) = 1/s, P(X = 0) = 1 - 1/s.
    """
    s = tf.cast(s, settings.float_type)
    if seed is None:
        return tf.where(tf.random_uniform([n], dtype=settings.float_type) <= 1./s,
                tf.random_normal([n], dtype=settings.float_type), tf.zeros([n], dtype=settings.float_type))
    else:
        return tf.where(stateless.stateless_random_uniform([n], dtype=settings.float_type, seed = seed) <= 1./s,
                stateless.stateless_random_normal([n], dtype=settings.float_type, seed = seed), tf.zeros([n], dtype=settings.float_type))


def lr_hadamard_prod_sparse(A, B, num_components, sparse_scale, seed = None):
    """
    Low-rank Hadamard product with Very Sparse Johnson Lindenstrauss Transform.
    An improvement on lowrank_hadamard_prod_subsample with small additional cost. 
    We use a variant of the Very Sparse method replacing the +-1 entries with standard Gaussians. 
    See:
        https://users.soe.ucsc.edu/~optas/papers/jl.pdf
        http://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf
    # Input
    :A: A [..., k1] tensor
    :B: A [..., k2] tensor
    # Output
    :C: A [..., num_components] tensor
    """
    batch_shape = tf.shape(A)[:-1]
    k1 = tf.shape(A)[-1]
    k2 = tf.shape(B)[-1]
    idx1 = tf.reshape(tf.range(k1, dtype=settings.int_type), [1, -1, 1])
    idx2 = tf.reshape(tf.range(k2, dtype=settings.int_type), [-1, 1, 1])
    
    combinations = tf.reshape(tf.concat([idx1 + tf.zeros_like(idx2), tf.zeros_like(idx1) + idx2], axis=2), [-1, 2])
    
    D = k1 * k2
    rand_matrix_size = D * num_components
    
    if sparse_scale == 'log':
        s = tf.cast(D, settings.float_type) / tf.log(tf.cast(D, settings.float_type))
    elif sparse_scale == 'sqrt':
        s = tf.sqrt(tf.cast(D, settings.float_type))

    R = tf.reshape(_draw_n_sparse_gaussian_samples(rand_matrix_size, s, seed = seed), [D, num_components])
    
    idx_result = tf.count_nonzero(R, axis=1) > 0
    idx_combined = tf.boolean_mask(combinations, idx_result, axis=0)
    n_nonzero = tf.shape(idx_combined)[0]
    A = tf.reshape(tf.gather(A, idx_combined[:,0], axis=-1), [-1, n_nonzero])
    B = tf.reshape(tf.gather(B, idx_combined[:,1], axis=-1), [-1, n_nonzero])
    C = A * B
    R_nonzero = tf.boolean_mask(R, idx_result, axis=0)
    C = tf.matmul(C, R_nonzero)    
    scale = tf.sqrt(s / tf.cast(num_components, settings.float_type))
    return scale * tf.reshape(C, tf.concat((batch_shape, [num_components]), axis=0))

import tensorflow as tf

from gpflow import settings
from gpflow.conditionals import base_conditional
import numpy as np
from tensorflow.contrib import stateless
import tensorflow as tf
from gpflow import settings
from gpflow.kullback_leiblers import gauss_kl


def draw_indices(n, l, need_inv = False):
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


def Nystrom_map(X, kern, num_components = None, inducing_samples = None, return_inducing_samples = False):
    """
    Computes the Nystrom features with uniform sampling given a kernel and num_components
    See e.g. https://dl.acm.org/citation.cfm?id=2343678
    -------------------------------------------------------------------
    Input
        :X:             input data points with size (num_samples, num_dims)
        :kern:          function handle to a kernel function that takes two matrices as input
                            e.g. X1 (num_samples1, num_dims) and X2 (num_samples2, num_dim), and computes
                            the matrix k(X1, X2) matrix of size (num_samples1, num_samples2)
        :num_components:number of components to take, i.e. the rank of the low-rank kernel matrix
    Output
        :X_nys:         Nystrom features of size (num_samples, num_components)
    """
    
    num_samples = tf.shape(X)[0]
    
    if inducing_samples is None:
        idx, idx_not, rev_map = draw_indices(num_samples, num_components, need_inv=True)
        X_sampled = tf.gather(X, idx, axis = - 2)
        X_not_sampled = tf.gather(X, idx_not, axis = - 2)
        W = kern(X_sampled, X_sampled) + tf.diag(settings.numerics.jitter_level * tf.random_uniform([num_components], dtype=settings.float_type))
        K21 = kern(X_not_sampled, X_sampled)
        C = tf.concat((W, K21), axis = - 2)
        S, U = tf.self_adjoint_eig(W)
        D = tf.sqrt(tf.maximum(tf.cast(settings.numerics.jitter_level, settings.float_type) ** 2, S))
        X_nys = tf.matmul(C,U) / tf.expand_dims(D, axis = -2)    
        X_nys = tf.gather(X_nys, rev_map, axis = - 2)
    else:
        num_components = tf.shape(inducing_samples)[0]
        W = kern(inducing_samples, inducing_samples) + tf.diag(settings.numerics.jitter_level * tf.random_uniform([num_components], dtype=settings.float_type))
        Kxy = kern(tf.reshape(X, [-1, tf.shape(X)[-1]]), inducing_samples)
        S, U = tf.self_adjoint_eig(W)
        D = tf.sqrt(tf.maximum(tf.cast(settings.numerics.jitter_level, settings.float_type) ** 2, S))
        X_nys = tf.matmul(Kxy, U) / tf.expand_dims(D, axis = -2)
        X_nys = tf.reshape(X_nys, tf.concat((tf.shape(X)[:-1], [num_components]), axis=0))
    if return_inducing_samples:
        return X_nys, X_sampled
    else:
        return X_nys


def lr_hadamard_prod(A, B):
    """
    Computes the low-rank equivalent of the Hadamard product between matrices.    
    Inputs
    :A: An [..., k1] tensor
    :B: An [..., k2] tensor
    Output
    :C: An [..., k1*k2] tensor
    """
    C = tf.matmul(tf.expand_dims(A, axis=-1), tf.expand_dims(B, axis=-2))
    return tf.reshape(C, tf.concat((tf.shape(C)[:-2], [tf.reduce_prod(tf.shape(C)[-2:], axis=0)]), axis=0))

def lr_hadamard_prod_rand(A, B, rank_bound, rand='sqrt', seeds=None):
    """
    Computes a randomized low-rank Hadamard product    
    Inputs
    :A:     An [..., k1] tensor
    :B:     An [..., k2] tensor
    :rand:  Randomization mode
    Output
    :C:     An [..., k1*k2] tensor
    """
    if rand == 'subsample':
        if seeds is None:
            C = lr_hadamard_prod_subsample(A, B, rank_bound)
        else:
            C = lr_hadamard_prod_subsample(A, B, rank_bound, seeds[i-1])
    elif rand == 'subsample_gauss':
        if seeds is None:
            C = lr_hadamard_prod_subsample_gauss(A, B, rank_bound)
        else:
            C = lr_hadamard_prod_subsample_gauss(A, B, rank_bound, seeds[i-1])
    else:
        if seeds is None:
            C = lr_hadamard_prod_sparse(A, B, rank_bound, sparsity)
        else:
            C = lr_hadamard_prod_sparse(A, B, rank_bound, sparsity, seeds[i-1])
    
    return C

def draw_n_rademacher_samples(n, seed = None):
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
    Inputs
    :A: An [..., k1] tensor
    :B: An [..., k2] tensor
    Output
    :return C: An [..., num_components] tensor
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
    D = tf.expand_dims(draw_n_rademacher_samples(num_components, seed = seed), axis=0)    
    return tf.reshape(C * D, tf.concat((batch_shape, [num_components]), axis=0))


def draw_n_gaussian_samples(n, seed = None):
    """
    Draws n gaussian samples.
    """
    if seed is None:
        return tf.random_normal([n], dtype=settings.float_type)
    else:
        return stateless.stateless_random_normal([n], dtype=settings.float_type, seed = seed)


def lr_hadamard_prod_subsample_gauss(A, B, num_components, seed = None):
    """
    Low-rank Hadamard product with subsampling.
    Inputs
    :A: An [..., k1] tensor
    :B: An [..., k2] tensor
    Output
    :return C: An [..., num_components] tensor
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
    
    D = 1./tf.sqrt(tf.cast(num_components, settings.float_type)) * \
        tf.reshape(draw_n_gaussian_samples(num_components**2, seed = seed), [num_components, num_components])    
    return tf.reshape(tf.matmul(C, D), tf.concat((batch_shape, [num_components]), axis=0))

def draw_n_sparse_gaussian_samples(n, s, seed = None):
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
    Inputs
    :A: An [..., k1] tensor
    :B: An [..., k2] tensor
    Output
    :C: An [..., num_components] tensor
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

    R = tf.reshape(draw_n_sparse_gaussian_samples(rand_matrix_size, s, seed = seed), [D, num_components])
    
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


def gauss_conditional_feature_simplify_maybe(phi, phi_, q_mu, q_sqrt, full_cov, white):
        num_samples, num_out_components = tf.unstack(tf.shape(phi))
        num_samples_ = tf.shape(phi_)[0]
        f_mean, f_var = tf.cond(num_out_components < num_samples + num_samples_,
                                lambda: gauss_conditional_feature_simplified(phi, phi_, white, q_mu, q_sqrt, full_cov),
                                lambda: gauss_conditional_feature(phi, phi_, white, q_mu, q_sqrt, full_cov))
        return f_mean, f_var

def gauss_conditional_feature(phi, phi_, white, f, q_sqrt=None, full_cov=False):

    num_samples, num_samples_ = tf.shape(phi)[0], tf.shape(phi_)[0]

    Kmm = tf.matmul(phi, phi, transpose_b=True) + (white + settings.numerics.jitter_level) * tf.eye(num_samples, dtype=settings.float_type)
    Kmn = tf.matmul(phi, phi_, transpose_b=True)
    if full_cov:
        Knn = tf.matmul(phi_, phi_, transpose_b=True) + white * tf.eye(num_samples_, dtype=settings.float_type)
    else:
        Knn = tf.reduce_sum(tf.square(phi_), axis=1) + white
    
    f_mean, f_var = base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt = tf.matrix_band_part(q_sqrt, -1, 0))
    return f_mean, f_var

def gauss_conditional_feature_simplified(phi, phi_, white, f, q_sqrt = None, full_cov = False):
    
    num_samples_, num_out_components = tf.unstack(tf.shape(phi_)) # N, K

    A = tf.matmul(phi, phi, transpose_a=True) + (white + settings.numerics.jitter_level) * tf.eye(num_out_components, dtype=settings.float_type)
    L = tf.cholesky(A) # K x K
    alpha = tf.matrix_transpose(tf.cholesky_solve(L, tf.matrix_transpose(phi_))) # N x K 
    beta = tf.matmul(alpha, phi, transpose_b=True) # N x M

    num_latent = tf.shape(f)[1] # L
    f_mean = tf.matmul(beta, f) # N x L

    if q_sqrt is not None:
        beta_tiled = tf.tile(tf.expand_dims(beta, axis=0), [num_latent, 1, 1]) # L x N x M
        beta_tiled_L = tf.matmul(beta_tiled, q_sqrt) # L x N x M

    if full_cov:
        f_var = white * (tf.eye(num_samples_, dtype=settings.float_type) + tf.matmul(alpha, phi_, transpose_b=True)) # N x N
        if q_sqrt is not None:
            f_var = tf.tile(tf.expand_dims(f_var, axis=0), [num_latent, 1, 1]) # L x N x N 
            f_var += tf.matmul(beta_tiled_L, beta_tiled_L, transpose_b=True) # L x N x N
    else:
        f_var = white * (1 + tf.reduce_sum(alpha * phi_, axis=1)) # N
        f_var = tf.tile(tf.expand_dims(f_var, axis=1), [1, num_latent]) # N x L
        if q_sqrt is not None:
            f_var += tf.matrix_transpose(tf.reduce_sum(tf.square(beta_tiled_L), axis=2))

    return f_mean, f_var

def gauss_kl_feature_simplified(phi, q_mu, q_sqrt, white):
    """
    Compute the KL divergence KL[q || p] between
          q(x) = N(q_mu, q_sqrt^2)
    and
          p(x) = N(0, phi phi.T + sigma^2 I)
    Comments from GPFlow:
        We assume N multiple independent distributions, given by the columns of
        q_mu and the last dimension of q_sqrt. Returns the sum of the divergences.
        q_mu is a matrix (N x L), each column contains a mean.
        q_sqrt is a 3D tensor (L x N x N), each matrix within is a lower
            triangular square-root matrix of the covariance of q.
        phi is an (N x K) low-rank factor of K up to a non-zero added diagonal
        white a scalar representing an isotropic diagonal matrix added to the kernel matrix
        (So K = phi phi.T + white I) 
    """
    
    num_latent = tf.shape(q_mu)[1] # L
    num_samples, num_out_components = tf.unstack(tf.shape(phi)) # M, K

    A = tf.matmul(phi, phi, transpose_a=True) + (white + settings.numerics.jitter_level) * tf.eye(num_out_components, dtype=settings.float_type) # K x K
    L_A = tf.cholesky(A) # K x K
    B  = tf.matrix_triangular_solve(L_A, tf.matrix_transpose(phi)) # K x M
    B_tiled = tf.tile(tf.expand_dims(B, axis=0), [num_latent, 1, 1]) # L x K x M

    trace = 1 / white * (tf.reduce_sum(tf.square(q_sqrt)) - tf.reduce_sum(tf.square(tf.matmul(B_tiled, q_sqrt)))) # scalar

    mahalanobis = 1 / white * (tf.reduce_sum(tf.square(q_mu)) - tf.reduce_sum(tf.square(tf.matmul(B, q_mu)))) # scalar

    num_latent, num_samples, num_out_components = tf.unstack(tf.cast((num_latent, num_samples, num_out_components), settings.float_type))
    const = -1. * num_latent * num_samples

    log_det = num_latent * (num_samples - num_out_components) * tf.log(white) + num_latent * tf.reduce_sum(tf.log(tf.square(tf.matrix_diag_part(L_A)))) \
            - tf.reduce_sum(tf.log(tf.square(tf.matrix_diag_part(q_sqrt)))) # scalar
    
    KL = 0.5 * (trace + mahalanobis + log_det + const)

    return KL    
    

def gauss_kl_feature_simplify_maybe(phi, q_mu, q_sqrt, white):
        num_out_components = tf.shape(phi)[-1]
        num_samples = tf.shape(phi)[0]
        KL = tf.cond(num_out_components < num_samples,
                    lambda: gauss_kl_feature_simplified(phi, q_mu, q_sqrt, white),
                    lambda: gauss_kl(q_mu, q_sqrt, tf.matmul(phi, phi, transpose_b=True) + 
                        (settings.numerics.jitter_level + white) * tf.eye(num_samples, dtype=settings.float_type)))
        return KL

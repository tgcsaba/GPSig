import sys
sys.path.append('..')

import numpy as np

from sklearn.model_selection import ParameterGrid
from gpsig import mmd_utils
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, laplacian_kernel
from gpsig.kernels import SignatureLinear, SignatureRBF, SignatureLaplace


SIG_LIN_PARAMS_GRID = ParameterGrid({'low_rank' : [True],
                                     'rank_bound' : [100],
                                     'num_levels' : [2, 3, 4],
                                     'add_time': [True],
                                     'normalize_kernel' : [True],
                                     'normalize_data' : [True],
                                     'normalize_levels' : [True, False]
                                    })

SIG_RBF_PARAMS_GRID = ParameterGrid({'low_rank' : [True],
                                     'rank_bound' : [100],
                                     'lengthscales' : np.logspace(-5, 5, 10),
                                     'num_levels' : [2, 3, 4],
                                     'add_time': [True],
                                     'normalize_kernel' : [True],
                                     'normalize_data' : [True],
                                     'normalize_levels' : [True, False]
                                    })


SIG_LAP_PARAMS_GRID = ParameterGrid({'low_rank' : [True],
                                     'rank_bound' : [100],
                                     'lengthscales' : np.logspace(-5, 5, 10),
                                     'num_levels' : [2, 3, 4],
                                     'add_time': [True],
                                     'normalize_kernel' : [True],
                                     'normalize_data' : [True],
                                     'normalize_levels' : [True, False]
                                    })

LIN_PARAMS_GRID = ParameterGrid({'normalize_kernel' : [True]})
RBF_PARAMS_GRID = ParameterGrid({'gamma' : np.logspace(-10, 1, 10)})
LAP_PARAMS_GRID = ParameterGrid({'gamma' : np.logspace(-10, 1, 10)})

def mmd_max_lin(X, Y):
    return mmd_utils.mmd_max(linear_kernel, X, Y, LIN_PARAMS_GRID, name='Lin', verbose=False)

def mmd_max_rbf(X, Y):
    return mmd_utils.mmd_max(rbf_kernel, X, Y, RBF_PARAMS_GRID, name='RBF', verbose=False)

def mmd_max_laplace(X, Y):
    return mmd_utils.mmd_max(laplacian_kernel, X, Y, LAP_PARAMS_GRID, name='Laplace', verbose=False)

def perm_test_lin(X, Y, num_permutations):
    return mmd_utils.mmd_max_permutation_test(linear_kernel, X, Y, LIN_PARAMS_GRID, num_permutations)

def perm_test_rbf(X, Y, num_permutations):
    return mmd_utils.mmd_max_permutation_test(rbf_kernel, X, Y, RBF_PARAMS_GRID, num_permutations)

def perm_test_laplace(X, Y, num_permutations):
    return mmd_utils.mmd_max_permutation_test(laplacian_kernel, X, Y, LAP_PARAMS_GRID, num_permutations)

def mmd_max_sig_lin(X, Y):
    return mmd_utils.mmd_max(SignatureLinear, X, Y, SIG_LIN_PARAMS_GRID, batch_size=200, name='LR-Sig-Lin', verbose=False)

def mmd_max_sig_rbf(X, Y):
    return mmd_utils.mmd_max(SignatureRBF, X, Y, SIG_RBF_PARAMS_GRID, batch_size=200, name='LR-Sig-RBF', verbose=False)

def mmd_max_sig_laplace(X, Y):
    return mmd_utils.mmd_max(SignatureLaplace, X, Y, SIG_LAP_PARAMS_GRID, batch_size=200, name='LR-Sig-Laplace', verbose=False)

def perm_test_sig_lin(X, Y, num_permutations):
    return mmd_utils.mmd_max_permutation_test(SignatureLinear, X, Y, SIG_LIN_PARAMS_GRID, num_permutations, batch_size=100)

def perm_test_sig_rbf(X, Y, num_permutations):
    return mmd_utils.mmd_max_permutation_test(SignatureRBF, X, Y, SIG_RBF_PARAMS_GRID, num_permutations, batch_size=100)

def perm_test_sig_laplace(X, Y, num_permutations):
    return mmd_utils.mmd_max_permutation_test(SignatureLaplace, X, Y, SIG_LAP_PARAMS_GRID, num_permutations, batch_size=100)
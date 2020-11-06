import sys
sys.path.append('..')

import numpy as np

from sklearn.model_selection import ParameterGrid

from gpsig.kernels import SignatureLinear, SignatureRBF, SignatureLaplace
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, laplacian_kernel

import tensorflow_datasets as tfds
import medical_ts_datasets

from get_data import data_UEA, data_medical

from functools import partial

# EXPERIMENTS:
NUM_PERMUTATIONS = 100 # for the permutation statistics
NUM_REPETITIONS = 25 # how often each experiment is repeated
NUM_SAMPLES = [30, 70, 200] # number of samples m
# NUM_SAMPLES = [30] # number of samples m
LEN_SEQUENCE = [10, 100, 200, 500]
# LEN_SEQUENCE = [10]
DIM_STATE_SPACE = [5000]
SIGNIFICANCE_LEVEL = 5

RESULTS_DIR = './results/'

# DATASETS:

def data_medical(name):
    data = [x for x in tfds.as_numpy(tfds.load(name, split='train'))]
    M = [x['combined'][3] for x in data]
    T = [x['combined'][1] for x in data]
    X = [x['combined'][2] for x in data]
    Y = [x['target'] for x in data]
    means = np.nanmean(np.concatenate(X, axis=0), axis=0)
    def impute(x, m):
        x[0, np.logical_not(m[0])] = means[np.logical_not(m[0])]
        for i in range(1, x.shape[0]):
            x[i, np.logical_not(m[i])] = x[i-1, np.logical_not(m[i])]
        return x
    X = [np.float64(impute(x, M[i])) for i, x in enumerate(X)]
    data = [np.concatenate((T[i][:, None], x), axis=1) for i, x in enumerate(X)]
    labels = [np.any(y==1) for y in Y]
    
    X = [data[i] for i, y in enumerate(labels) if y==0]
    Y = [data[i] for i, y in enumerate(labels) if y==1]
            
    return X, Y

# DATASETS_UEA = ['ArticularyWordRecognition',
#                 'AtrialFibrillation',
#                 'BasicMotions',
#                 'CharacterTrajectories',
#                 'Cricket',
#                 'DuckDuckGeese',
#                 'ERing',
#                 'EigenWorms',
#                 'Epilepsy',
#                 'EthanolConcentration',
#                 'FaceDetection',
#                 'FingerMovements',
#                 'HandMovementDirection',
#                 'Handwriting',
#                 'Heartbeat',
#                 'Images',
#                 'InsectWingbeat',
#                 'JapaneseVowels',
#                 'LSST',
#                 'Libras',
#                 'MotorImagery',
#                 'NATOPS',
#                 'PEMS-SF',
#                 'PenDigits',
#                 'PhonemeSpectra',
#                 'RacketSports',
#                 'SelfRegulationSCP1',
#                 'SelfRegulationSCP2',
#                 'SpokenArabicDigits',
#                 'StandWalkJump',
#                 'UWaveGestureLibrary']

DATASETS_UEA = ['EthanolConcentration',
                'PhonemeSpectra',
                'HandMovementDirection',
                'SelfRegulationSCP2',
                'SelfRegulationSCP1',
                'Epilepsy',
                'DuckDuckGeese',
                'AtrialFibrillation',
                'FaceDetection',
                'StandWalkJump',
                'MotorImagery',
#                 'InsectWingbeat',
#                 'PEMS-SF',
                'PenDigits',
                'FingerMovements',
                'EigenWorms',            
                'Heartbeat']


DATASETS_MEDICAL = ['physionet2012',
                    'physionet2019']

DATASETS = {**{dataset : partial(data_UEA, dataset) for dataset in DATASETS_UEA}, **{dataset : partial(data_medical, dataset) for dataset in DATASETS_MEDICAL}}
    

# KERNELS:
SIG_LIN_PARAMS_GRID = ParameterGrid({'low_rank' : [True],
                                     'rank_bound' : [100],
                                     'num_levels' : [2, 3, 4, 5],
                                     'add_time': [False],
                                     'normalize_kernel' : [True],
                                     'normalize_data' : [True],
                                     'normalize_levels' : [False]
                                    })

SIG_RBF_PARAMS_GRID = ParameterGrid({'low_rank' : [True],
                                     'rank_bound' : [100],
                                     'lengthscales' : np.logspace(-5, 5, 10),
                                     'num_levels' : [2, 3, 4, 5],
                                     'add_time': [False],
                                     'normalize_kernel' : [True],
                                     'normalize_data' : [True],
                                     'normalize_levels' : [False]
                                    })


SIG_LAP_PARAMS_GRID = ParameterGrid({'low_rank' : [True],
                                     'rank_bound' : [100],
                                     'lengthscales' : np.logspace(-5, 5, 10),
                                     'num_levels' : [2, 3, 4, 5],
                                     'add_time': [False],
                                     'normalize_kernel' : [True],
                                     'normalize_data' : [True],
                                     'normalize_levels' : [False]
                                    })

LIN_PARAMS_GRID = ParameterGrid({'normalize_kernel' : [True],
                                 'normalize_data' : [True]})

RBF_PARAMS_GRID = ParameterGrid({'gamma' : np.logspace(-10, 10, 10),
                                 'normalize_kernel' : [True],
                                 'normalize_data' : [True]
                                })

LAP_PARAMS_GRID = ParameterGrid({'gamma' : np.logspace(-10, 10, 10),
                                 'normalize_kernel' : [True],
                                 'normalize_data' : [True]
                                })

SIG_BATCH_SIZE = 50
VEC_BATCH_SIZE = 500

KERNELS = {'Linear' : (linear_kernel, LIN_PARAMS_GRID, VEC_BATCH_SIZE),
           'RBF' : (rbf_kernel, RBF_PARAMS_GRID, VEC_BATCH_SIZE),
           'Laplace' : (laplacian_kernel, LAP_PARAMS_GRID, VEC_BATCH_SIZE),
           'SignatureLinear' : (SignatureLinear, SIG_LIN_PARAMS_GRID, SIG_BATCH_SIZE),
           'SignatureRBF' : (SignatureRBF, SIG_RBF_PARAMS_GRID, SIG_BATCH_SIZE),
           'SignatureLaplace' : (SignatureLaplace, SIG_LAP_PARAMS_GRID, SIG_BATCH_SIZE)
          }


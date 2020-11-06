"""
Benchmark Two-sample tests for time series data

H0: \mu \eq \nu
H1: \mu \neq \nu

where \mu,\nu are the probability measure of multi-variate time-series

for each dataset repeat the following repetition(=100) times:
    
Uniformly select m samples from (X,Y) under H0
Uniformly select m samples from (X,Y) under H1

Process the each time series as follows:

* Random time change until they have the same length
    If it's longer than L trunacte at length TS_max_len
    If it's shorter than L randomly select a sequence index and double it, repeat until it's of length TS_max_len

** Maximal state space dimension
    If the state space of the time-series is bigger than TS_max_len use only the first TS_max_len entries

We denote the resulting data sample_0 and sample_0
each is a collection of m time series X and m time series Y

Do a permutation test for each test statistic:
    calculate T(X,Y) and compare it to the distribution of T(pi(X,Y)) when pi is a uniformly at random choosen permutation
    if T(X,Y) is inside the 95% quantile accept, otherwise reject

Do the above for various values of m, TS_max_len, TS_max_dim to study the influence of sample size, time series length, and state space dimension

Datasets:
We use labelled time series data
We only use two different labels, say {A,B}, and ignore the rest of the data if there are more than two labels
Under H0 both \mu and \nu are the distribution of time series labelled A
Under H1 both \mu and \nu are the distribution of time series labelled B
"""


import os
import sys

if len(sys.argv) > 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(sys.argv[1])
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import random
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from settings import *
import pickle
import time

from get_data import list2array, pad_with_time_change
from gpsig.mmd_utils import mmd_max_permutation_test


exp_params = [(m, TS_max_dim, TS_max_len) for m in NUM_SAMPLES for TS_max_dim in DIM_STATE_SPACE for TS_max_len in LEN_SEQUENCE]

for dataset_name in DATASETS:
    
    print("Checking results for dataset: {}...".format(dataset_name))
    
    load_dataset = False
    for kernel_name in KERNELS:
        for m, TS_max_dim, TS_max_len in exp_params:
            result_file = os.path.join(RESULTS_DIR, kernel_name, '{}_M{}_L{}_D{}.pkl'.format(dataset_name, m, TS_max_len, TS_max_dim))
            if not os.path.exists(result_file):
                print('Missing result file: {}...'.format(result_file))
                load_dataset = True

    print('Loading dataset: {}...'.format(dataset_name))
    
    # get a list of numpy arrays
    # one list entry is a time series
    X_all, Y_all = DATASETS[dataset_name]()
    

    X_samples, Y_samples = len(X_all), len(Y_all)

    print('X:{} | Y:{}'.format(X_samples, Y_samples))
    #remove nan's in case there are any
    X_all = [ A[~np.isnan(A).any(axis=1)] for A in X_all]
    Y_all = [ A[~np.isnan(A).any(axis=1)] for A in Y_all]
    
    for kernel_name in KERNELS:
        
        print('Kernel: {}...'.format(kernel_name))
        
        kernel, params_grid, batch_size = KERNELS[kernel_name]

        for m, TS_max_dim, TS_max_len in exp_params:

            # choose number of samples, sequence length, state space dimension
            print('Samples: ', m, 'Dimension <= ', TS_max_dim, 'Length <= ', TS_max_len, '...')
            
            if not os.path.isdir(os.path.join(RESULTS_DIR, kernel_name)):
                os.makedirs(os.path.join(RESULTS_DIR, kernel_name))
            
            run_file = os.path.join(RESULTS_DIR, kernel_name, '{}_M{}_L{}_D{}.txt'.format(dataset_name, m, TS_max_len, TS_max_dim))
            result_file = os.path.join(RESULTS_DIR, kernel_name, '{}_M{}_L{}_D{}.pkl'.format(dataset_name, m, TS_max_len, TS_max_dim))
            
            if os.path.exists(run_file):
                print('Found run file: {}...'.format(run_file))
                continue
                
            with open(run_file, 'wb') as f:
                pass
              
            result_dict = {'H0' : [], 'H1' : []}
            if os.path.exists(result_file):
                try:
                    with open(result_file, 'rb') as f:
                        result_dict = pickle.load(f)
                except:
                    pass
                    
                
            if min(len(result_dict['H0']), len(result_dict['H1'])) >= NUM_REPETITIONS:
                continue            
            
            starting_repetition = min(len(result_dict['H0']), len(result_dict['H1']))

            #truncate every ts at max length and max state space dimension
            X_all_sub = [ts[0:TS_max_len, 0:TS_max_dim] for ts in X_all]
            Y_all_sub = [ts[0:TS_max_len, 0:TS_max_dim] for ts in Y_all]
            
            # repeat: select m samples from X, Y under H0 and under H1
            # add random time change to pad to same length
            for repetition in range(starting_repetition, NUM_REPETITIONS):
                start = time.time()
                print('Repetition: ', repetition+1, '/', NUM_REPETITIONS, '...')

                #select randomly (uniformly with replacement) some ts for the experiment
                X, Y, Z = random.choices(X_all_sub, k=m), random.choices(Y_all_sub, k=m), random.choices(X_all_sub, k=m)

                # perturb with random time change
                length = 2 * max([max([ts.shape[0] for ts in A]) for A in [X, Y, Z]])
#                 length = 2 * TS_max_len
                state_space_dimension = X[0].shape[1]
                X_padded, Y_padded, Z_padded = pad_with_time_change(X, length), pad_with_time_change(Y, length), pad_with_time_change(Z, length)    
                print('Random time change produces sequences of length {} evolving in {} coordinates.'.format(length, state_space_dimension))

                if 'Signature' not in kernel_name:
                    # for non-signature kernels data is just a (samples, dim) array
                    # turn list of np.arrays into (samples, flattened-dim) numpy array
#                     X_, Y_, Z_  = list2array(X_padded), list2array(Y_padded), list2array(Z_padded)
                    X_, Y_, Z_  = X_padded, Y_padded, Z_padded
#                     X_, Y_, Z_ = X, Y, Z

                else:
                    X_, Y_, Z_ = X, Y, Z


                # H0 same distributions mu=nu
                # H1 different distributions mu != nu
                hypothesis=dict([('H0', (X_,Z_)), ('H1', (X_,Y_))])

                for hyp in hypothesis:

#                     experiment = (dataset_name, kernel_name, hyp, repetition)
                    
                    print('Hypothesis: {}...'.format(hyp))

                    A, B = hypothesis[hyp][0],  hypothesis[hyp][1]

                    try:
                        hist, statistic_eval = mmd_max_permutation_test(kernel, A, B, params_grid, NUM_PERMUTATIONS, batch_size=batch_size)
                    except:
                        continue
                    

                    perc = np.percentile(hist, 100 - SIGNIFICANCE_LEVEL)

                    reject = statistic_eval > perc
                    result_dict[hyp].append((statistic_eval, hist, SIGNIFICANCE_LEVEL, int(reject)))
                    
                    if hyp == 'H0': 
                        # both sets of samples come from same distribution
                        if not reject:
                            print('Success: H0 (mu=nu) and {} is not larger than 95\% percentile {}.'.format(statistic_eval, perc))
                        else:
                            print('Failure: H0 (mu=nu) but {} is larger than 95\% percentile {}.'.format(statistic_eval, perc))
                    elif hyp =='H1':
                        # the sets of samples come from different distributions
                        if not reject:
                            print('Failure: H1 (mu != nu) but {} is not larger than 95\% percentile {}.'.format(statistic_eval, perc))
                        else:
                            print('Success: H1 (mu != nu) and {} is larger than 95\% percentile {}.'.format(statistic_eval, perc))

                print("finished in:",(time.time() - start), "seconds\n")
        
            with open(result_file, 'wb') as f:
                pickle.dump(result_dict, f)
            os.remove(run_file)
            print('Saved result file: {}'.format(result_file))


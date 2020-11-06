#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark Two-sample tests for time series data

H0: \mu \neq \nu
H1: \mu \eq \nu

where \mu,\nu are the laws of time-series

fix a test statistic (WaldWolfowitz, Hotelling, MMDs)

repeat the following:

for \mu=\nu 50 samples X from \mu and 50 samples Y from \nu 
the same for \mu \neq \nu

do a permutation test: calculate T(X,Y) and compare it to the distribution of T(pi(X,Y)) 
when pi is a uniformly at random choosen permutation

if T(X,Y) is inside the 95% quantile reject etc


For the generation of samples we use labelled time series data
we use two different labels (ignore the rest if there are more than 2)
choose 50 time series for each label
then in each repeition we pad them randomly to turn into same lenght so that they can be vectorized
(maybe change this later)

"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
import matplotlib.pyplot as plt
from get_data import data_randomwalk, data_shiftedNormal, data_signal, list2array, data_UEA, multTS, multTS_hard, pad_with_time_change, subsample_datapoints
from hotelling import t_statistic
from WaldWolfowitz import ww_test
from classic_kernels import rbf_mmd, laplace_mmd
from functools import partial
import os
import pandas as pd
import time

from kernels_new import mmd_max_lin, mmd_max_rbf, mmd_max_laplace, perm_test_lin, perm_test_rbf, perm_test_laplace, mmd_max_sig_lin, mmd_max_sig_rbf, mmd_max_sig_laplace, perm_test_sig_lin, perm_test_sig_rbf, perm_test_sig_laplace

from gpsig.preprocessing import interp_list_of_sequences

from scipy.spatial.distance import squareform, pdist, cdist


plot_folder=os.getcwd()+'/plots/'
TS_max_len=100
TS_max_dim=5
TS_max_size=2

def sq_distances(X,Y=None):
    """
    If Y=None, then this computes the distance between X and itself
    """
    assert(X.ndim==2)

    # IMPLEMENT: compute pairwise distance matrix. Don't use explicit loops, but the above scipy functions
    # if X=Y, use more efficient pdist call which exploits symmetry
    if Y is None:
        sq_dists = squareform(pdist(X, 'sqeuclidean'))
    else:
        assert(Y.ndim==2)
        assert(X.shape[1]==Y.shape[1])
        sq_dists = cdist(X, Y, 'sqeuclidean')

    return sq_dists


def two_sample_permutation_test(test_statistic, X, Y, num_permutations):
    """
    Parameters
    ----------
    test_statistic : a function that that takes two arrays of same dimensions 
                    as X,Y as input and returns a real number
    
    X, Y : numpy array of same dimensions
    
    num_permutations : number of permutations used

    Returns
    -------
    a (num_permuations,) numpy array where the i-th entry is the test_statistic 
   evaluated at two arrays stacked together and permutated randomly the first index

    """
    assert X.ndim == Y.ndim
    
    statistics = np.zeros(num_permutations)
    
    range_ = range(num_permutations)
    
    for i in range_:
        # concatenate samples
        if X.ndim == 1:
            Z = np.hstack((X,Y))
        elif X.ndim == 2:
            Z = np.vstack((X,Y))
        elif X.ndim == 3:
            Z = np.concatenate((X, Y), axis=0)
            
        # IMPLEMENT: permute samples and compute test statistic
        perm_inds = np.random.permutation(len(Z))
        Z = Z[perm_inds]
        X_ = Z[:len(X)]
        Y_ = Z[len(X):]
        statistics[i] = test_statistic(X_, Y_)
        
    return statistics

def plot_permutation_samples(null_samples, title, statistic=None):
    plt.hist(null_samples)
    percentile_low, percentile_high = np.percentile(null_samples, 2.5), np.percentile(null_samples, 97.5)
    plt.axvline(x=percentile_low, c='b')
    legend = ["95% quantiles"]
    if statistic is not None:
        plt.axvline(x=statistic, c='r')
        legend += ["Actual test statistic"]
    plt.legend(legend)
    plt.axvline(x=percentile_high, c='b')
    plt.xlabel("Test statistic value")
    plt.ylabel("Counts")
    plt.title(title)
    plt.savefig(plot_folder+title+'.pdf')
    plt.close('all')
    return percentile_low, percentile_high

if __name__ == '__main__':
    multTS=multTS_hard

    repetitions = 100
    
    datasets_UEA = dict([(ts, partial(data_UEA, ts)) for ts in multTS])

    datasets_synthetic = dict([('synthetic signal', data_signal), 
                 ('synthetic random walk', data_randomwalk),
                 ('synthetic shifted normal', data_shiftedNormal)])
    
    datasets_all = {**datasets_synthetic, **datasets_UEA}
    
    statistics = dict([('Hotelling t statistic', (t_statistic,)),
                       ('Wald-Wolfowitz statistic', (ww_test,)),
                       ('MMD Lin', (mmd_max_lin, perm_test_lin)),
                       ('MMD RBF', (mmd_max_rbf, perm_test_rbf)),
                       ('MMD Laplace', (mmd_max_laplace, perm_test_laplace)),
                       ('MMD Sig Lin', (mmd_max_sig_lin, perm_test_sig_lin)),
                       ('MMD Sig RBF', (mmd_max_sig_rbf, perm_test_sig_rbf)),
                       ('MMD Sig Lap', (mmd_max_sig_laplace, perm_test_sig_laplace))
                       ])

    num_permutations=100
    datasets = datasets_synthetic
    datasets = datasets_UEA
    
    result_keys = [data +','+ stat +','+ hyp for data in datasets 
                                           for stat in statistics 
                                           for hyp in ['H0','H1']
                                           #for val in ['lower percenticle', 'higher percentile', 'statistic']
                                           ]
    
    df_experiments = pd.DataFrame(None)
    results=dict([(k,np.nan) for k in result_keys])
    
    experiments= [(dataset, statistic, hypothesis, rep) for dataset in datasets.keys() for statistic in statistics.keys() for hypothesis in ['H0','H1'] for rep in range(repetitions)] 
    df_ind=pd.MultiIndex.from_tuples(experiments, names=('Dataset', 'Statistic', 'Hypothesis', 'Repetition'))
    df_columns=['Success', 'X samples', 'Y samples', 'Statistic', 'Percentile low', 'Percentile high', '(Padded) Length', 'Dimension']
    df=pd.DataFrame(None, index=df_ind, columns = df_columns)
    df=df.sort_index()
    for dataset in datasets:
        
        print("Dataset: ", dataset)
        
        #get list of numpy arrays
        X_all, Y_all, U_all, V_all = datasets[dataset]()
        
        X_all=[ts[0:TS_max_len,0:TS_max_dim] for ts in X_all]
        Y_all=[ts[0:TS_max_len,0:TS_max_dim] for ts in Y_all]
        U_all=[ts[0:TS_max_len,0:TS_max_dim] for ts in U_all]
        V_all=[ts[0:TS_max_len,0:TS_max_dim] for ts in V_all]

        
        #remove nan's in case there are any
        X_all = [ A[~np.isnan(A).any(axis=1)] for A in X_all]
        Y_all = [ A[~np.isnan(A).any(axis=1)] for A in Y_all]
        U_all = [ A[~np.isnan(A).any(axis=1)] for A in U_all]
        V_all = [ A[~np.isnan(A).any(axis=1)] for A in V_all]
        
        for repetition in range(repetitions):
            start = time.time()
            print('Repetition', repetition+1, '/',repetitions)
            
            X = subsample_datapoints(X_all, TS_max_size)
            Y = subsample_datapoints(Y_all, TS_max_size)
            U = subsample_datapoints(U_all, TS_max_size)
            V = subsample_datapoints(V_all, TS_max_size)
            
            # perturb with random time change
            # length =  2*max([max([ts.shape[0] for ts in A]) for A in [X, Y, U, V]])
            state_space_dimension=X[0].shape[1]
            # X_padded, Y_padded, U_padded, V_padded = pad_with_time_change(X, length), pad_with_time_change(Y, length), pad_with_time_change(U, length), pad_with_time_change(V, length) 
            
            X_samples, Y_samples, U_samples, V_samples = len(X), len(Y), len(U), len(V)
            length =  max([max([ts.shape[0] for ts in A]) for A in [X, Y, U, V]])
            X_padded, Y_padded, U_padded, V_padded = np.split(interp_list_of_sequences(X+Y+U+V), [X_samples, X_samples+Y_samples, X_samples+Y_samples+U_samples])
            
            #turn list of np.arrays into (samples, flattened-dim) numpy array
            X_array, Y_array  = list2array(X_padded), list2array(Y_padded)
            U_array, V_array  = list2array(U_padded), list2array(V_padded)
    
            
            for statistic in statistics:
                
                #print("Statistic: ",statistic)
                
                if 'MMD Sig' not in statistic:
                    #for non-signature kernels data is just a (samples, dim) array
                    X_,Y_ = X_array, Y_array
                    U_,V_ = U_array, V_array
                else:
                    X_,Y_ = X,Y
                    U_,V_ = U,V
                   
                
                s=statistics[statistic][0]
                
                hypothesis=dict([('H0', (X_,Y_)), ('H1', (U_,V_))])
                
                for hyp in hypothesis:
                    experiment=(dataset,statistic,hyp,repetition)
                    print('Experiment:', experiment)
                    
                    A, B = hypothesis[hyp][0],  hypothesis[hyp][1]
                    
                    if len(statistics[statistic]) == 1:
                        hist=two_sample_permutation_test(s, A,B, num_permutations)
                        statistic_eval = s(A,B)   
                    else:
                        hist,statistic_eval=statistics[statistic][1](A,B, num_permutations)   
                        
                    # uncomment to save a plot of permuation
                    # plt_title= dataset+' with '+statistic + ' under ' + hyp
                    # perc_Low, perc_High = plot_permutation_samples(hist,plt_title, statistic_eval)
                   
                    perc_Low, perc_High = np.percentile(hist, 2.5), np.percentile(hist, 97.5)
                    
          
                    if hyp=='H0':
                        if not(perc_Low < statistic_eval < perc_High):
                            print("Success: H0 and ",statistic_eval,' is not between percentiles (',perc_Low, ',',perc_High,')')
                            success=True
                        else:
                            print("Failure: H0 but ",statistic_eval,' is between percentiles (',perc_Low, ',',perc_High,')')
                            success=False
                        
                    if hyp =='H1':
                        if (perc_Low < statistic_eval < perc_High):
                            print("Success: H1 and ",statistic_eval,' is between percentiles (',perc_Low, ',',perc_High,')')
                            success=True
                        else:
                            print("Failure: H1 but ",statistic_eval,' is not between percentile (',perc_Low, ',',perc_High,')')
                            success=False
                    
                    df.loc[experiment,['Success','X samples', 'Y samples', 'Statistic', 'Percentile low', 'Percentile high', '(Padded) Length', 'Dimension']]=[success,len(A),len(B), statistic_eval, perc_Low, perc_High, length, state_space_dimension]
                    #df.loc[experiment,['Success','X samples', 'Y samples', 'Statistic', 'Percentile low', 'Percentile high', '(Padded) Length', 'Dimension']]=[success,A.shape[0],B.shape[0], [repetition,repetition*2], perc_Low, perc_High, length, state_space_dimension]
            print("finished in:",(time.time() - start), "seconds\n")
        
            
        df.to_pickle("./df_results.pkl")  
        print("saved to pkl:",dataset)
        
    #pd.read_pickle("df_results.pk")
 #alt=df.loc[('EthanolConcentration', 'Hotelling t statistic', 'H1'),'Statistic']  
 #plt.hist([list(null),list(alt)],label=['H0', 'H1']), plt.legend(loc='upper right')
# df.loc[('EthanolConcentration', 'Hotelling t statistic', 'H1'),['Percentile low','Dimension']]
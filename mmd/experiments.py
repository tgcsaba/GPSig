#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
import random
import matplotlib.pyplot as plt
from get_data import data_randomwalk, data_shiftedNormal, data_signal, list2array, data_UEA, multTS_hard, pad_with_time_change
from hotelling import t_statistic
from WaldWolfowitz import ww_test
from classic_kernels import rbf_mmd, laplace_mmd
from functools import partial
import os
import pandas as pd
import time

from kernels_new import mmd_max_lin, mmd_max_rbf, mmd_max_laplace, perm_test_lin, perm_test_rbf, perm_test_laplace, mmd_max_sig_lin, mmd_max_sig_rbf, mmd_max_sig_laplace, perm_test_sig_lin, perm_test_sig_rbf, perm_test_sig_laplace


from scipy.spatial.distance import squareform, pdist, cdist


plot_folder=os.getcwd()+'/plots/'

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
    
    num_permutations=100 #for the permutation statistics
    repetitions = 25 #how often each experiment is repeated
    number_samples = [30, 70, 200] #number of samples  m
    sequence_length = [10,100,200,500]
    dimension = [5000]
    
    experiment_parameters = [ (m, TS_max_dim, TS_max_len) for m in number_samples for TS_max_dim in dimension for TS_max_len in sequence_length]
    
    #select datasets
    # datasets_UEA = dict([ (ts, data_UEA(ts)) for ts in multTS_hard])
    # datasets_synthetic = dict([('synthetic signal', data_signal), 
    #                            ('synthetic random walk', data_randomwalk),
    #                            ('synthetic shifted normal', data_shiftedNormal)])
    # datasets_all = {**datasets_synthetic, **datasets_UEA}
    # datasets = datasets_UEA #just do UEA
    datasets = multTS_hard
    
    #select test statistics
    statistics = dict([
                       # ('Hotelling t statistic', (t_statistic,)),
                       # ('Wald-Wolfowitz statistic', (ww_test,)),
                       ('MMD Lin', (mmd_max_lin, perm_test_lin)),
                       ('MMD RBF', (mmd_max_rbf, perm_test_rbf)),
                       ('MMD Laplace', (mmd_max_laplace, perm_test_laplace)),
                       # ('MMD Sig Lin', (mmd_max_sig_lin, perm_test_sig_lin)),
                       # ('MMD Sig RBF', (mmd_max_sig_rbf, perm_test_sig_rbf)),
                       # ('MMD Sig Lap', (mmd_max_sig_laplace, perm_test_sig_laplace))
                       ])

    
    
    #setup experiments and prepare df to store results in
    experiments = [(data, statistic, param) for data in datasets for statistic in statistics.keys() for param in experiment_parameters] 
    
    df_ind=pd.MultiIndex.from_tuples(experiments, names=('Dataset', 'Statistic', 'Parameters'))
    df_columns=['false reject', 'false accept']
    
    df=pd.DataFrame(None, index=df_ind, columns = df_columns)
    df=df.sort_index()
    df[:]=0
        
    #select what is recorded for each experiment
    result_keys = [data +','+ stat +','+ hyp for data in datasets 
                                           for stat in statistics 
                                           for hyp in ['H0','H1']
                                           #for val in ['lower percenticle', 'higher percentile', 'statistic']
                                           ]
    
   # df_experiments = pd.DataFrame(None)
    results=dict([(k,np.nan) for k in result_keys])
    
    
    
    for dataset in datasets:
        
        print("Dataset: ", dataset)
        
        #get a list of numpy arrays
        #one list entry is a time series
        X_all, Y_all = data_UEA(dataset)
        
        
        X_samples, Y_samples = len(X_all), len(Y_all)
        
        
        #remove nan's in case there are any
        X_all = [ A[~np.isnan(A).any(axis=1)] for A in X_all]
        Y_all = [ A[~np.isnan(A).any(axis=1)] for A in Y_all]
        
        for params in experiment_parameters:
            
            m, TS_max_dim, TS_max_len = params
          
            
            #choose number of samples, sequence length, state space dimension
            print('Samples',m,'Dimension <=', TS_max_dim, 'Length', TS_max_len)
            
            
            #truncate every ts at max length and max state space dimension
            X = [ts[0:TS_max_len,0:TS_max_dim] for ts in X_all]
            Y = [ts[0:TS_max_len,0:TS_max_dim] for ts in Y_all]
   
            
            #repeat: select m samples from X,Y under H0 and under H1
            #add random time change to pad to same length
            for repetition in range(repetitions):
                start = time.time()
                print('Repetition', repetition+1, '/',repetitions)
                
                
                #select randomly (uniformly with replacement) some ts for the experiment
                X, Y, Z = random.choices(X, k=m), random.choices(Y, k= m), random.choices(X, k=m)
            
                # perturb with random time change
                length =  2*max([max([ts.shape[0] for ts in A]) for A in [X,Y,Z]])
                state_space_dimension=X[0].shape[1]
                X_padded, Y_padded, Z_padded = pad_with_time_change(X, length), pad_with_time_change(Y, length), pad_with_time_change(Z, length)      
                print('Random time change produces sequences of length', length, 'evolving in', state_space_dimension,'coordinates')
               
        
                
                for statistic in statistics:

                    
                    if 'MMD Sig' not in statistic:
                        #for non-signature kernels data is just a (samples, dim) array
                        #turn list of np.arrays into (samples, flattened-dim) numpy array
                        X_, Y_, Z_  = list2array(X_padded), list2array(Y_padded), list2array(Z_padded)
                        
                    else:
                        X_,Y_ = X,Y,Z
                       
                    
                    s=statistics[statistic][0]
                    
                    #H0 same distributions mu=nu
                    #H1 different distributions mu != nu
                    hypothesis=dict([('H0', (X_,Z_)), ('H1', (X_,Y_))])
                    
                    for hyp in hypothesis:
                        
                        experiment=(dataset,statistic,hyp,repetition)
                        print('Experiment:', experiment)
                        
                        A, B = hypothesis[hyp][0],  hypothesis[hyp][1]
                        
                        if len(statistics[statistic]) == 1:
                            #huh: Csaba why this?
                            hist=two_sample_permutation_test(s, A,B, num_permutations)
                            statistic_eval = s(A,B)   
                        else:
                            hist,statistic_eval=statistics[statistic][1](A,B, num_permutations)   
                            
                        # uncomment to save a plot of permuation statistic
                        # plt_title= dataset+' with '+statistic + ' under ' + hyp
                        # perc_Low, perc_High = plot_permutation_samples(hist,plt_title, statistic_eval)
                       
                        perc_Low, perc_High = np.percentile(hist, 2.5), np.percentile(hist, 97.5)
                        
                        
                        if hyp=='H0': 
                            #both sets of samples come from same distribution 
                            if not(perc_Low <= statistic_eval <= perc_High):
                                print("Success: H0 (mu=nu) and ",statistic_eval,' is in percentiles (',perc_Low, ',',perc_High,')')
                            else:
                                print("Failure: H0 (mu=nu) but ",statistic_eval,' is not in percentiles (',perc_Low, ',',perc_High,')')
                                df.loc[(dataset, statistic, params)]['false reject']+= 1.0

                        if hyp =='H1':
                            #the sets of samples come from different distributions
                            if (perc_Low <= statistic_eval <= perc_High):
                                print("Failure: H1 (mu != nu) but ",statistic_eval,' is in percentiles (',perc_Low, ',',perc_High,')')
                            else:
                                print("Success: H1 (mu != nu) and ",statistic_eval,' is not in percentiles (',perc_Low, ',',perc_High,')')
                                df.loc[(dataset, statistic, params)]['false accept']+= 1.0
                            

                               # df.loc[(dataset, params, statistic),['H0 falsely accepted']] +=1
                       
                        # [success,A.shape[0],B.shape[0], [repetition,repetition*2], perc_Low, perc_High, length, state_space_dimension]
                print("finished in:",(time.time() - start), "seconds\n")
                
        df.to_pickle("./df_results.pkl")  #save after each dataset in case sth crashes

    df = df.div(repetitions)
    df.to_pickle("./df_results.pkl")  
    print("saved to pkl:",dataset)
    
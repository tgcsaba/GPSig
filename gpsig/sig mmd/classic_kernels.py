#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 15:20:50 2020

@author: harald
"""
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel
import numpy as np


def quadratic_time_mmd(K_XX,K_XY,K_YY):

    n = len(K_XX)
    m = len(K_YY)
    
    # IMPLEMENT: unbiased MMD statistic (could also use biased, doesn't matter if we use permutation tests)
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    mmd = np.sum(K_XX) / (n*(n-1))  + np.sum(K_YY) / (m*(m-1))  - 2*np.sum(K_XY)/(n*m)
    return mmd

def mmd_optimized(X,Y, kernel, verbose= False):
    """
    

    Parameters
    ----------
    
    X : (m_samples, dim) numpy array
    Y : (n_samples, dim) numpy array
    kernel: fuction that evaluates X,Y,g to a matrix

    Returns
    -------
    (m_samples,n_samples) numpy array
    
    """
    k_opt=-1.0
    g_opt=-1.0
    g_min, g_max = 10**(-10), 10**1
    
    for g in np.logspace(-10, 10, 20):
        K_XX = kernel(X,X,g)
        K_XY = kernel(X,Y,g)
        K_YY = kernel(Y,Y,g)
        
        k_new= quadratic_time_mmd(K_XX,K_XY,K_YY)
        
        if k_new > k_opt:
            k_opt = k_new
            g_opt = g
            
    if verbose:
        if g_opt == g_min or g_opt == g_max:
            print("parameter on boundary ", g_opt)
        else:
            print("parameter inside", g_opt)
    return k_opt, g_opt

def rbf_mmd(X,Y):
    return mmd_optimized(X, Y, rbf_kernel)[0]

def laplace_mmd(X,Y):
    return mmd_optimized(X, Y, laplacian_kernel)[0]
    




if __name__ == '__main__':
    X=np.random.rand(1000,30)
    Y=np.random.rand(2000,30)
    print(rbf_mmd(X,Y))

    
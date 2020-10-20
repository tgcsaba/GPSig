#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:34:06 2020


"""
import numpy as np
import matplotlib.pyplot as plt


def t_statistic(X,Y):
    """
    Parameters
    ----------
    X, : (m,dim) numpy array
         data consisting of m samples in dimension dim.
  

    Returns 
    
    the Hotelling two-sample T^2 statistic 
    t^2 = mn/(m+n) (mean(X)-mean(Y))^T S^{-1} (mean(X)-mean(Y))
    -------
    None.
    """
    mu=X.mean(0)-Y.mean(0)
    m,n =X.shape[0], Y.shape[0]
    Sigma =( (m-1)*np.cov(X.T, bias=False) + (n-1)*np.cov(Y.T, bias=False) ) / (m+n-2)
#    print(Sigma)
    Sigma_inv= np.linalg.pinv(Sigma)
    return m*n/(m+n) * ( mu @ (Sigma_inv @ mu) )



if __name__ == '__main__':
    samples=[0]*500
    for i in range(500):
        X=np.random.normal(0,1,(1400,2))
        Y=np.random.normal(0.5,1,(1450,2))
        samples[i]=t_statistic(X, Y)
    plt.hist(samples)
    plt.show()
    
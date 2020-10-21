#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synthetic data 
"""

import numpy as np

def signal(ticks,spins,eps,sigma):
    """
    

    Parameters
    ----------
    ticks : number of time samples
    spins : TYPE
        DESCRIPTION.
    eps : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.

    Returns
    -------
    (ticks,2) numpy array. The ticks are the times time series
    f(i)= (eps*sin(2*i*pi*spins),eps*cos(2*i*pi*spins))+N_i(0,sigma)

    
    """
    
    f=np.asarray([[eps*np.sin(x*2*np.pi*spins), eps*np.cos(x*2*np.pi*spins)] for x in np.linspace(0,1,ticks)])
    mean = np.random.normal(0,5,2)
    return f+np.random.normal(mean, sigma, (ticks,2))

# def signal2(ticks,spins,eps,sigma):
#     """
#     returns a list containing ticks numpy arrays of shape (2,). 
#     the i-the element of the list represent the i-th tick of the time series
#     f(i)= (eps*sin(2*i*pi*spins),eps*cos(2*i*pi*spins))+N_i(0,sigma)
#     """
#     f=np.asarray([[np.sin(x*2*np.pi*spins),np.cos(x*2*np.pi*spins)] for x in np.linspace(0,1,ticks)])
#     f=eps*f
#     mean = np.random.normal(0,5,2)
#     return f+np.random.normal(mean, sigma, (ticks,2))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    s=signal(100,10, 0.02, 0.01)
    s_x=[i[0] for i in s]
    s_y=[i[1] for i in s]
    
    plt.plot(s_x,s_y, linestyle='--', marker='o')
    plt.plot(s_x[0], s_y[0], 'ob', markersize = 13)
    plt.plot(s_x[-1], s_y[-1], 'ob', fillstyle='none', markersize=13)
    plt.savefig('signal_trajectory', format = 'pdf')
    plt.xlabel('dimension 1')
    plt.ylabel('dimension 2')
    plt.show()
    
    
    
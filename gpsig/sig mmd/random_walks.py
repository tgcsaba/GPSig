#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 16:30:43 2018

@author: oberhauser
"""
import numpy as np

def rw(samples, ticks, dimension):
    """
    returns list (samples). The i-th entry represents a sequence as a (ticks, dim) array
    Each sequence starts at 0 and is simple lattice random walk
    """
    ONB = np.eye(dimension)
    increments = np.vstack((ONB,-ONB))
    origin = np.zeros(dimension)
    
    R = [None] * samples 
    np.empty((samples, ticks, dimension))
    for s in range(samples):    
        #generates array of ticks-1 integers in [0,...,dimension-1]
        #note tick = 0 will start at the origin
        rnd = np.random.randint(0,increments.shape[0], ticks-1) 
        R[s]=np.cumsum(np.asarray([origin]+[increments[r,:] for r in rnd]), axis =0)
    return R

def fake_rw1d(ticks, window):
    """
    generates sequence of Bernoulli random variables of length ticks
    replaces every (window+1)-th element by the product of window entries before
    """
    incs=2*(np.random.randint(0,2, ticks)-0.5)
    for i in range(1,ticks//window+1):
        incs[i*window-1]= np.prod(incs[i*window-window:i*window-1])
    r=np.cumsum(incs)
    r=np.roll(r,1)
    r[0]=0.0
    return r

def rw1d(ticks):
    """
    generates sequence of Bernoulli random variables of length ticks
    returns their cumsum
    """
    r=np.cumsum(2*(np.random.randint(0,2, ticks)-0.5))
    r=np.roll(r,1)
    r[0]=0.0
    return r


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.clf()

    for i in range(30):
        s=rw1d(100)
        plt.plot(s)
        plt.xlabel('time')
#    plt.show()
    plt.savefig('rw100ticks.pdf', format = 'pdf')
    plt.clf()

#    plt.plot(s_x[0], s_y[0], 'ob', markersize = 13)
#    plt.plot(s_x[-1], s_y[-1], 'ob', fillstyle='none', markersize=13)
  #  plt.savefig('signal_trajectory', format = 'pdf')
   
    for i in range(30):
        s=fake_rw1d(100,3)
        plt.xlabel('time')
        plt.plot(s)
 #   plt.show()
    plt.savefig('Fakerw100ticks.pdf', format = 'pdf')
        
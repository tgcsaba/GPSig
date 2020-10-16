#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_arff.zip
extract and put into folder /Multivariate_arff/ in same directory where get_data.py is located
"""
import numpy as np
from signal_vs_noise import signal
from random_walks import rw1d,fake_rw1d
from sktime.utils.load_data import load_from_arff_to_dataframe
from sklearn.model_selection import StratifiedShuffleSplit
import os


DATA_PATH_UEA=os.getcwd()+'/Multivariate_arff/'

multTS=[
         'SpokenArabicDigits',
         'Cricket',
          'BasicMotions',
          #'PEMS-SF',
          'EthanolConcentration',
          'PhonemeSpectra',
          'Handwriting',
          'Libras',
          'HandMovementDirection',
          'SelfRegulationSCP2',
          'CharacterTrajectories',
          'SelfRegulationSCP1',
          'ERing',
          'Epilepsy',
          'DuckDuckGeese',
          'LSST',
          'AtrialFibrillation',
          'FaceDetection',
          'StandWalkJump',
          'MotorImagery',
       #   'InsectWingbeat',
          'PenDigits',
          'FingerMovements',
          'EigenWorms',
         # 'Images',
         'NATOPS',
         'JapaneseVowels',
         'UWaveGestureLibrary',
         'RacketSports',
         'ArticularyWordRecognition',
         'Heartbeat']


#Subset of the above which seem to make some tests struggle
multTS_hard=[
          'EthanolConcentration',
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
       #   'InsectWingbeat',
          'PenDigits',
          'FingerMovements',
          'EigenWorms',            
         'Heartbeat']

   

def data_UEA(name, N):
    """


    Parameters
    ----------
    dataset_name : string
        DESCRIPTION. One of the names of the time series specified in multTS
    
    N: number of time series 
    
    null: Boolean, optional
        DESCRIPTION. if null==True then all time series returned have the same label (0)
                     if null==False then X contains time series labelled 0 and Y the ones labelled 1
        

    Returns
    -------
    X, Y : list of length N (or less if less than N samples in dataset) 
                the i-th list entry is a (len_i,dim) numpy array
        DESCRIPTION. X are the TS labelled with 0, Y are the TS labelled with 1   
    U, V :

    """
    
 
    print("importing dataset")
    X_train, y_train = load_from_arff_to_dataframe(os.path.join(DATA_PATH_UEA, './{0}/{0}_TRAIN.arff'.format(name)))
    X_test, y_test = load_from_arff_to_dataframe(os.path.join(DATA_PATH_UEA, './{0}/{0}_TEST.arff'.format(name)))
    print("import finished")
   
    
    labels_dict = {c : i for i, c in enumerate(np.unique(y_train))}
    
    #transform into one big labelled dataset 
    labels = [labels_dict[c] for c in y_train] + [labels_dict[c] for c in y_test]
    data = [np.stack(x, axis=1) for x in X_train.values] + [np.stack(x, axis=1) for x in X_test.values]
    print("number of time-series:", len(labels), "with", len(np.unique(labels)), "labels" )
    
        
    X=[data[i] for i in range(len(labels)) if labels[i]==0]
    Y=[data[i] for i in range(len(labels)) if labels[i]==1]
    
    
    data=[data[i] for i in range(len(labels)) if (labels[i]==0 or labels[i]==1)]
    labels = [labels[i] for i in range(len(labels)) if (labels[i]==0 or labels[i]==1)]
    sss= StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    labels_tr, labels_te = next(sss.split(data, labels))
    U=[data[i] for i in labels_tr]
    V=[data[i] for i in labels_te]

    return X[:N], Y[:N], U[:N], V[:N]

def time_change(X, new):
    """
    
    given a sequence of "length" many vectors pick randomly a sequence entry and double it
    repeat "new"-many times to turn into sequence of length+new many vectors

    Parameters
    ----------
    x : (length, dimension) numpy array
    new : int how many new time entries are added
    

    Returns
    -------
    (length + new , dimensions) numpy array

    """
    A=X
    length=X.shape[0]
    for r in range(new):
        ind=np.random.randint(0,length+r)
        A=np.insert(A, ind, A[ind,:], 0) 
    
    return A


def data_signal(N, null=True):
    """
        Parameters
    ----------
    N : samples
    null : Boolean
        DESCRIPTION. The default is True.

    Returns
    -------
    X, Y: list of length N of (ticks,2) numpy arrays
  
    """
    ticks, spins, eps, sigma = 100, 10, 0.03, 0.5
    #samples_x, samples_y = 50, 50

    X=[signal(ticks, spins, eps, sigma) for n in range(N)]

    if null==True:
        Y=[ signal(ticks, spins, eps, sigma) for n in range(N)]
        
    else:
        Y=[ signal(ticks, spins, 0.0, sigma) for n in range(N)]
      

    return X,Y

def data_randomwalk(N, null=True):
    """
    
TODO: returns 1dim rw
    Parameters
    ----------
    N : number of time series
    null : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    X,Y: lists of (100,1) numpy arrays

    """

    ticks, window = 100, 3
    #samples_x, samples_y = 50, 50
    
    X=[ rw1d(ticks) for n in range(N)]
    
    if null==True:
        Y=[rw1d(ticks) for n in range(N)]
   
    else:   
        Y=[fake_rw1d(ticks, window) for n in range(N)]
        
    X=[np.asarray([ts]).transpose() for ts in X ]
    Y=[np.asarray([ts]).transpose() for ts in Y ]

        
    return X,Y

def data_shiftedNormal(N, null=True):
    """
    

    Parameters
    ----------
    N : nr of samples
    null : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    X, Y : list of length N of (1,2) nump arrays
        each entry is a 2-dim gaussian with same variance and mean shifted for Y if null is false

    """
   
    mean, var=0.0, 1.0
    
    if null==True:
        X, Y = [np.random.normal(mean, var, (1,2)) for n in range(N)], [np.random.normal(mean, var, (1,2)) for n in range(N)]
        #X,Y = np.random.normal(mean,var,(N,2)), np.random.normal(mean,var,(N,2))
    else:
        X,Y = [np.random.normal(mean, var, (1,2)) for n in range(N)], [np.random.normal(mean+0.5, var, (1,2)) for n in range(N)]
        #X,Y = np.random.normal(mean,var,(N,2)), np.random.normal(mean+0.5,var,(N,2))
        
    
  
    return X,Y

def list2array(X):
    """
    
    Parameters
    ----------
    X : list of numpy arrays
        DESCRIPTION.

    Returns
    -------
    None.
    
    """
    
    return np.asarray([np.ndarray.flatten(x) for x in X])



def pad_with_time_change(X, length_max):
    """
    

    Parameters
    ----------
    X : list of numpy arrays
        we assume the i-th entry is of shape (len_i, dim). That is the dim is the same for all entries.

    Returns
    -------
    X_padded : list of numpy arrays of shape (len, dim) where len = max len_i
        DESCRIPTION.

    """
    
    X_padded = [None]*len(X)
    for i,ts in enumerate(X):
        length = ts.shape[0]
        X_padded[i]=time_change(ts, length_max - length)
    return X_padded
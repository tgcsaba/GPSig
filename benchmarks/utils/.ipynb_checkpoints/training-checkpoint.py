import sys
import os
sys.path.append('..')

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import gpflow as gp
import gpsig

import keras
from keras import backend as K

def fit_nn_with_gp_layer(X_train, y_train, var_list, x_tens, y_tens, loss, sess, minibatch_size=50, max_epochs=1000,
                         val_scores=None, lower_is_better=True, patience=None, history=None, w_tens=None, W_train=None):
    
    assert (w_tens is None) == (W_train is None)

    opt = keras.optimizers.Adam(clipvalue=5.)
    
    param_updates = opt.get_updates(loss, var_list)

    num_train = X_train.shape[0]
    
    num_batches_per_epoch = int(np.ceil(float(num_train) / minibatch_size))
    
    train_phase = True
    
    if w_tens is None:
        train_batch = K.function(inputs=[K.learning_phase(), x_tens, y_tens], outputs=[loss], updates=param_updates)
    else:
        train_batch = K.function(inputs=[K.learning_phase(), x_tens, y_tens, w_tens], outputs=[loss], updates=param_updates)
    
    if history is None:
        history = {}
        start_epoch = 0
        if val_scores is not None:
            best_score = np.inf if lower_is_better else -np.inf
            num_epochs_since_best = 0
    else:
        prev_epochs = [k for k in history.keys() if isinstance(k, int)]
        start_epoch = np.max(prev_epochs) + 1 if len(prev_epochs) > 0 else 0
        if val_scores is not None:
            params_saved = []
            for var in var_list:
                params_saved.append(sess.run(var))
            history['best'] = {}
            history['epoch'] = start_epoch
            for i, scorer in enumerate(val_scores):
                _score = scorer()
                history['best']['val_{}'.format(i)] = _score
            history['best']['params'] = params_saved
            best_score = _score
            num_epochs_since_best = 0
    
    for epoch in range(start_epoch, max_epochs + start_epoch):
        if patience is not None and num_epochs_since_best > patience:
            break
        
        inds = np.random.permutation(X_train.shape[0])
        l_avg = 0.
        for t in range(num_batches_per_epoch):
            X_batch = X_train[inds[t*minibatch_size:np.minimum(num_train, (t+1)*minibatch_size)]]
            y_batch = y_train[inds[t*minibatch_size:np.minimum(num_train, (t+1)*minibatch_size)], None]
            if w_tens is None:
                l = train_batch([train_phase, X_batch, y_batch, None])[0]
            else:
                W_batch = W_train[inds[t*minibatch_size:np.minimum(num_train, (t+1)*minibatch_size)]]
                l = train_batch([train_phase, X_batch, y_batch, W_batch])[0]
                
            print('\rEpoch: {0:04d}/{1:04d} | Batch {2:2d}/{3:2d} | ELBO: {4:.3f}'.format(epoch+1, max_epochs + start_epoch, t+1, num_batches_per_epoch, -l), end='')
            l_avg += l
        l_avg /= float(num_batches_per_epoch)
        print('\rEpoch: {0:04d}/{1:04d} | Batch {2:2d}/{2:2d} | ELBO: {3:.3f} '.format(epoch+1, max_epochs + start_epoch, num_batches_per_epoch, -l_avg), end='')
            
        history[epoch] = {}
        history[epoch]['elbo'] = -l_avg
        
        if val_scores is not None:
            for i, scorer in enumerate(val_scores):
                _score = scorer()
                history[epoch]['val_{}'.format(i)] = _score
                print('| Val.{}.: {:.3f} '.format(i, _score), end='')
        
            if lower_is_better and _score <= best_score or not lower_is_better and _score >= best_score:
                best_score = _score
                num_epochs_since_best = 0
                params_saved = []
                for var in var_list:
                    params_saved.append(sess.run(var))
                history['best'] = {}
                history['epoch'] = epoch
                for key, val in history[epoch].items():
                    history['best'][key] = val
                history['best']['params'] = params_saved
                print('| New best...', end='')
            else:
                num_epochs_since_best += 1
        
        print()
        
    return history
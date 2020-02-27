import os
import sys

sys.path.append('..')

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import gpflow as gp
import gpsig

import pickle
import matplotlib.pyplot as plt

import keras
from keras import backend as K

from utils import *

from sklearn.metrics import accuracy_score, classification_report

def train_gprnn_classifier(dataset, num_hidden=128, num_inducing=500, normalize_data=True, minibatch_size=50, rnn_type='lstm', use_dropout=True,
                           val_split=None, test_split=None, experiment_idx=None, save_dir='./GPRNN/'):
    

    rnn_type = rnn_type.lower()
    if rnn_type not in ['lstm', 'gru']: raise ValueError('rnn_type should be \'LSTM\' or \'GRU\'')

    print('##########################################################################')
    print('Training dataset: {} | RNN = {}, num_hidden = {}, use_dropout = {}'.format(dataset, rnn_type, num_hidden, int(use_dropout)))
    print('##########################################################################')
    
    ## load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset, val_split=val_split, test_split=test_split,
                                                                  normalize_data=normalize_data, add_time=False, for_model='nn')
    
    num_train, len_examples, num_features = X_train.shape
    num_val = X_val.shape[0] if X_val is not None else None
    num_test = X_test.shape[0]
    num_classes = np.unique(y_train).size
    
    with tf.Session(graph=tf.Graph(), config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        
        ## setup NN part of model
        K.set_floatx('float64')
        x_tens = K.placeholder(shape=(None, len_examples, num_features), dtype=gp.settings.float_type)
        y_tens = K.placeholder(shape=(None, 1), dtype=gp.settings.float_type)

        masking_layer = keras.layers.Masking(mask_value=0., input_shape=(len_examples, num_features))

        recurrent_dropout = 0.05 if use_dropout else 0.
        dropout = 0.25 if use_dropout else 0.
        
        if rnn_type == 'lstm':
            rnn_layer = keras.layers.LSTM(num_hidden, recurrent_dropout=recurrent_dropout, dropout=dropout)
        elif rnn_type == 'gru':
            rnn_layer = keras.layers.GRU(num_hidden, recurrent_dropout=recurrent_dropout, dropout=dropout)
            
        fx_tens = rnn_layer(masking_layer(x_tens))

        nn_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, rnn_type)

        ## setup GP part of model    
        if num_classes == 2:
            lik = gp.likelihoods.Bernoulli()
            num_latent = 1
        else:
            lik = gp.likelihoods.MultiClass(num_classes)
            num_latent = num_classes

        Z = np.random.randn(num_inducing, num_hidden)

        k = gp.kernels.RBF(num_hidden, ARD=True)
        m = gp.models.SVGP(fx_tens, y_tens, kern=k, likelihood=lik, Z=Z, minibatch_size=None, num_latent=num_latent, num_data=num_train)

        # some tensorflow magic
        loss = - m.likelihood_tensor
        fmean, fvar = m._build_predict(fx_tens)
        ymean, yvar = m.likelihood.predict_mean_and_var(fmean, fvar)

        lpd = m.likelihood.predict_density(fmean, fvar, y_tens)
    
        K.set_session(sess)
        sess.run(tf.variables_initializer(var_list=nn_params))
    
        ## setup metrics
        if num_classes == 2:
            acc = lambda X, y: accuracy_score(y, sess.run(ymean, feed_dict={x_tens:X}).flatten() > 0.5)
        else:
            acc = lambda X, y: accuracy_score(y, np.argmax(sess.run(ymean, feed_dict={x_tens:X}), axis=1))

        nlpp = lambda X, y: -np.mean(sess.run(lpd, feed_dict={x_tens:X, y_tens:y[:, None]}))

        val_acc = lambda: acc(X_val, y_val)
        val_nlpp = lambda: nlpp(X_val, y_val)

        ## initalize inducing points with RNN-images of random data examples
        Z = sess.run(fx_tens, feed_dict={x_tens:X_train[np.random.choice(num_train, size=num_inducing)]})
        Z += 0.4 * np.random.randn(*Z.shape)
        m.feature.Z = Z

        ## initialize lengthscales parameter of RBF kernel
        fX_samples = sess.run(fx_tens, feed_dict={x_tens:X_train[np.random.choice(num_train, size=(np.minimum(1000, num_train)), replace=False)]})
        l = suggest_initial_lengthscales(fX_samples)
        m.kern.lengthscales = l

        ## train_model
        minibatch_size = np.minimum(50, num_train)

        ### phase 1 - pre-train variational distribution
        m.kern.set_trainable(False)
        hist = fit_nn_with_gp_layer(X_train, y_train, m.trainable_tensors, x_tens, y_tens, loss, sess, minibatch_size=minibatch_size, max_epochs=500)

        ### phase 2 - train full model
        m.kern.set_trainable(True)
        var_list = nn_params + m.trainable_tensors
        history = fit_nn_with_gp_layer(X_train, y_train, var_list, x_tens, y_tens, loss, sess, minibatch_size=minibatch_size, max_epochs=5000, 
                         val_scores=[val_acc, val_nlpp], patience=500, lower_is_better=True, history=hist)

        ### restore best parameters    
        for i, var in enumerate(var_list):
            sess.run(var.assign(history['best']['params'][i]))

        ### evaluate on validation data
        val_nlpp = val_nlpp()
        val_acc = val_acc()

        print('Val. nlpp: {:.4f}'.format(val_nlpp))
        print('Val. accuracy: {:.4f}'.format(val_acc))

        ### phase 3 - fix NN and kernel params and train on all of training data to assimilate into variational approximation
        #### re-merge validation data into training data
        if val_split is not None:
            X_train, y_train = np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0)
            num_train = X_train.shape[0]
            m.num_data = num_train

        #### train variational distribution 
        m.kern.set_trainable(False)
        history = fit_nn_with_gp_layer(X_train, y_train, m.trainable_tensors, x_tens, y_tens, loss, sess, minibatch_size=minibatch_size, max_epochs=500, history=hist)

        ## evaluate model
        num_iters = int(np.ceil(num_test / minibatch_size))
        y_test_pred = np.zeros((num_test), dtype=np.float64)
        y_test_nlpp = np.zeros((num_test), dtype=np.float64)
        for i in range(num_iters):
            print('\rComputing predictions... batch {}/{}'.format(i+1, num_iters), end='')
            X_current = X_test[i*minibatch_size:np.minimum((i+1)*minibatch_size, num_test)]
            y_current = y_test[i*minibatch_size:np.minimum((i+1)*minibatch_size, num_test)].reshape([-1, 1])
            if num_classes == 2:
                y_test_pred[i*minibatch_size:np.minimum((i+1)*minibatch_size, num_test)] = sess.run(ymean, feed_dict={x_tens:X_current}).flatten() > 0.5
            else:
                y_test_pred[i*minibatch_size:np.minimum((i+1)*minibatch_size, num_test)] = np.argmax(sess.run(ymean, feed_dict={x_tens:X_current}), axis=1)

            y_test_nlpp[i*minibatch_size:np.minimum((i+1)*minibatch_size, num_test)] = sess.run(lpd, feed_dict={x_tens:X_current, y_tens:y_current}).flatten()

        print()

        ## save results to file
        test_nlpp = -np.mean(y_test_nlpp)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_report = classification_report(y_test, y_test_pred)

        print('Test nlpp: {:.4f}'.format(test_nlpp))
        print('Test accuracy: {:.4f}'.format(test_acc))
        print(test_report)

        history['results'] = {}
        history['results']['val_acc'] = val_acc
        history['results']['val_nlpp'] = val_nlpp
        history['results']['test_nlpp'] = test_nlpp
        history['results']['test_acc'] = test_acc
        history['results']['test_report'] = test_report
        
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        experiment_name = '{}_H{}_D{}'.format(dataset, num_hidden, int(use_dropout))
        if experiment_idx is not None:
            experiment_name += '_{}'.format(experiment_idx)
        with open(os.path.join(save_dir, experiment_name + '.pkl'), 'wb') as f:
            pickle.dump(history, f)
        with open(os.path.join(save_dir, experiment_name + '.txt'), 'w') as f:
            f.write('Val. nlpp: {:.4f}\n'.format(val_nlpp))
            f.write('Val. accuracy: {:.4f}\n'.format(val_acc))
            f.write('Test. nlpp: {:.4f}\n'.format(test_nlpp))
            f.write('Test accuracy: {:.4f}\n'.format(test_acc))
            f.write('Test report:\n')
            f.write(test_report)
    
    # clear memory manually
    gp.reset_default_session()
    tf.reset_default_graph()
    K.clear_session()

    import gc
    gc.collect()

    return
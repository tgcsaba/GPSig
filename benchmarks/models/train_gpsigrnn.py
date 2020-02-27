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

def train_gpsigrnn_classifier(dataset, num_hidden=128, num_levels=4, num_inducing=500, normalize_data=True, minibatch_size=50, rnn_type='lstm', use_dropout=True,
                              max_len=500, increments=True, learn_weights=False, num_lags=None, val_split=None, test_split=None, experiment_idx=None, save_dir='./GPSigRNN/'):
    
    
    num_lags = num_lags or 0
    
    rnn_type = rnn_type.lower()
    if rnn_type not in ['lstm', 'gru']: raise ValueError('rnn_type should be \'LSTM\' or \'GRU\'')

    print('##########################################################################')
    print('Training dataset: {} | RNN = {}, num_hidden = {}, use_dropout = {}'.format(dataset, rnn_type, num_hidden, int(use_dropout)))
    print('##########################################################################')
    
    ## load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset, val_split=val_split, test_split=test_split, max_len=max_len,
                                                                  normalize_data=normalize_data, add_time=True, for_model='nn')
    
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
            rnn_layer = keras.layers.LSTM(num_hidden, recurrent_dropout=recurrent_dropout, dropout=dropout, return_sequences=True)
        elif rnn_type == 'gru':
            rnn_layer = keras.layers.GRU(num_hidden, recurrent_dropout=recurrent_dropout, dropout=dropout, return_sequences=True)
        
        # this is a hack to make reshape work with masking
        identity_layer = keras.layers.Lambda(lambda x: x, output_shape=lambda s: s)
        
        gp_input_dim = len_examples * num_hidden
        reshape_layer = keras.layers.Reshape((gp_input_dim,))
            
        fx_tens = reshape_layer(identity_layer(rnn_layer(masking_layer(x_tens))))

        nn_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, rnn_type)

        ## setup GP part of model    
        if num_classes == 2:
            lik = gp.likelihoods.Bernoulli()
            num_latent = 1
        else:
            lik = gp.likelihoods.MultiClass(num_classes)
            num_latent = num_classes

        # this is just temporary
        Z_init = np.random.randn(int(num_levels*(num_levels+1)/2), num_inducing, 2, num_hidden * (num_lags+1))
        feat = gpsig.inducing_variables.InducingTensors(Z_init, num_levels=num_levels, increments=increments, learn_weights=learn_weights)
        k = gpsig.kernels.SignatureRBF(gp_input_dim, num_levels=num_levels, num_features=num_hidden, num_lags=num_lags)
        
        if num_classes == 2:
            lik = gp.likelihoods.Bernoulli()
            num_latent = 1
        else:
            lik = gp.likelihoods.MultiClass(num_classes)
            num_latent = num_classes
        
        m = gpsig.models.SVGP(fx_tens, y_tens, kern=k, feat=feat, likelihood=lik, num_latent=num_latent, minibatch_size=None, num_data=num_train)
        
        # some tensorflow magic
        loss = - m.likelihood_tensor
        fmean, fvar = m._build_predict(fx_tens)
        ymean, yvar = m.likelihood.predict_mean_and_var(fmean, fvar)

        lpd = m.likelihood.predict_density(fmean, fvar, y_tens)
    
        K.set_session(sess)
        sess.run(tf.variables_initializer(var_list=nn_params))
    
        ## setup metrics
        def batch_predict_y(X, batch_size=None):
            num_iters = int(np.ceil(X.shape[0] / batch_size))
            y_pred = np.zeros((X.shape[0]), dtype=np.float64)
            for i in range(num_iters):
                slice_batch = slice(i*batch_size, np.minimum((i+1)*batch_size, X.shape[0]))
                X_batch = X[slice_batch]
                pred_batch = sess.run(ymean, feed_dict={x_tens:X_batch})
                if pred_batch.shape[1] == 1:
                    y_pred[slice_batch] = pred_batch.flatten() > 0.5
                else:
                    y_pred[slice_batch] = np.argmax(pred_batch, axis=1)
            return y_pred

        def batch_predict_density(X, y, batch_size=None):
            num_iters = int(np.ceil(X.shape[0] / batch_size))
            y_nlpp = np.zeros((X.shape[0]), dtype=np.float64)
            for i in range(num_iters):
                slice_batch = slice(i*batch_size, np.minimum((i+1)*batch_size, X.shape[0]))
                X_batch = X[slice_batch]
                y_batch = y[slice_batch, None]
                y_nlpp[slice_batch] = sess.run(lpd, feed_dict={x_tens:X_batch, y_tens:y_batch}).flatten()
            return y_nlpp

        acc = lambda X, y: accuracy_score(y, batch_predict_y(X, batch_size=minibatch_size))
        nlpp = lambda X, y: -np.mean(batch_predict_density(X, y, batch_size=minibatch_size))

        val_acc = lambda: acc(X_val, y_val)
        val_nlpp = lambda: nlpp(X_val, y_val)
        
        test_acc = lambda: acc(X_test, y_test)
        test_nlpp = lambda: nlpp(X_test, y_test)

        ## initalize inducing points with RNN-images of random data examples
        fX_samples = sess.run(fx_tens, feed_dict={x_tens:X_train[np.random.choice(num_train, size=num_inducing)]})
        fX_samples = fX_samples.reshape([-1, len_examples, num_hidden])
        Z_init = suggest_initial_inducing_tensors(fX_samples, num_levels, num_inducing, increments=increments, num_lags=num_lags)
        m.feature.Z = Z_init

        ## initialize lengthscales parameter of RBF kernel
        fX_samples = sess.run(fx_tens, feed_dict={x_tens:X_train[np.random.choice(num_train, size=(np.minimum(1000, num_train)), replace=False)]})
        fX_samples = fX_samples.reshape([-1, len_examples, num_hidden])
        l_init = suggest_initial_lengthscales(fX_samples, num_samples=1000)
        m.kern.lengthscales = l_init

        ## train_model
        minibatch_size = np.minimum(50, num_train)
    
        ### phase 1 - pre-train variational distribution
        m.kern.set_trainable(False)
        hist = fit_nn_with_gp_layer(X_train, y_train, m.trainable_tensors, x_tens, y_tens, loss, sess, minibatch_size=minibatch_size, max_epochs=500)
        
        
        ### phase 3 - train model with unfixed sigma_i
        m.kern.variances.set_trainable(True)
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

        ### phase 4 - fix NN and kernel params and train on rest of data to assimilate into variational approximation
        #### re-merge validation data into training data
        if val_split is not None:
            X_train, y_train = np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0)
            num_train = X_train.shape[0]
            m.num_data = num_train

        #### train variational distribution 
        m.kern.set_trainable(False)
        history = fit_nn_with_gp_layer(X_train, y_train, m.trainable_tensors, x_tens, y_tens, loss, sess, minibatch_size=minibatch_size, max_epochs=500, history=hist)

        ## evaluate model
        test_nlpp = test_nlpp()
        test_acc = test_acc()
        test_report = classification_report(y_test, batch_predict_y(X_test, batch_size=minibatch_size))

        print('Test nlpp: {:.4f}'.format(test_nlpp))
        print('Test accuracy: {:.4f}'.format(test_acc))
        print(test_report)
        
        ## save results to file
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
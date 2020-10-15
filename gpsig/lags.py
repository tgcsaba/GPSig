import numpy as np
import tensorflow as tf
from gpflow import config



def lin_interp(time, X, time_query):
    """
    Performs linear interpolation in time.
    # Input
    :time:          time points with shape (len_examples) 
    :X:             X values of shape (num_examples, len_examples, num_features)
    :time_query:    query time points with shape (len_examples, num_lags)
    # Output
    :X_query:       interpolated values of shape (num_examples, len_examples, num_lags, num_features)
    """

    len_examples, num_lags = tf.shape(time_query)[-2], tf.shape(time_query)[-1]
    
    pairwise_dist = time[:, None, None] - time_query[None, :, :]

    left_idx = tf.argmax(tf.where(pairwise_dist > config.default_jitter(), - np.inf * tf.ones_like(pairwise_dist), pairwise_dist), axis = 0)
    right_idx = left_idx + 1

    X_left = tf.gather(X, left_idx, axis = -2)
    X_right = tf.gather(X, right_idx, axis = -2)

    t_left = tf.gather(time, left_idx, axis = 0)
    t_right = tf.gather(time, right_idx, axis = 0)

    if X.shape.ndims == 3:
        X_query = X_left + (time_query[None, ..., None] - t_left[None, ..., None]) * (X_right - X_left) / (t_right[None, ..., None] - t_left[None, ..., None])
    elif X.shape.ndims == 4:
        X_query = X_left + (time_query[None, None, ..., None] - t_left[None, None, ..., None]) * (X_right - X_left) / (t_right[None, None, ..., None] - t_left[None, None, ..., None])
    else:
        raise ValueError('lags.lin_interp_time: Oops, X should either have ndims==3 or ndims==4.')

    return X_query


def add_lags_to_sequences(X, lags):
    """
    Given input sequences X adds its lagged versions as extra dimensions,
    where non-integer lagged versions are computed via linear interpolation.
    # Input
    :X:         an array of sequences of shape (..., num_examples, len_examples, num_features)
    :lags:      a vector of size (num_lags) containing the lag values 
    # Output
    :X_lags:    an array of shape (..., num_examples, len_examples, (num_lags + 1) * num_features)
    """

    num_examples, len_examples, num_features = tf.unstack(tf.shape(X))
    
    num_lags = tf.shape(lags)[0]

    time = tf.range(tf.cast(len_examples, config.default_float()), dtype=config.default_float()) / tf.cast(len_examples-1, config.default_float())
    time_lags = tf.maximum(time[:, None] - lags[None, :], 0.)

    X_lags = lin_interp(time, X, time_lags)
        
    X_new = tf.concat((X[:, :, None, :], X_lags), axis=2)

    return X_new

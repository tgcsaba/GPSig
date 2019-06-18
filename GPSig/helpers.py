import tensorflow as tf
from gpflow import settings
import numpy as np

######################
## Helper functions ##
######################

def lin_interp_time(time, X, time_query):
    """
    Performs linear interpolation in time.
    Inputs
    :time:  a vector of time points with size (streams_length)
    :X:     an array of shape (num_paths, streams_length, num_features)
    :time_query:  an array with lagged time points with size (streams_length, num_lags)
    Return
    :X_query: an array of shape (num_paths, streams_length, num_lags, num_features)
    """
    streams_length = tf.shape(time)[0]
    num_lags = tf.shape(time_query)[1]
    
    pairwise_dist = time[:, None, None] - time_query[None, :, :] 
    left_idx = tf.argmax(tf.where(pairwise_dist > 0, - np.inf * tf.ones_like(pairwise_dist), pairwise_dist), axis = 0)
    right_idx = left_idx + 1

    X_left = tf.gather(X, left_idx, axis = 1)
    X_right = tf.gather(X, right_idx, axis = 1)
    t_left = tf.reshape(tf.gather(time, left_idx, axis = 0), [1, streams_length, num_lags, 1])
    t_right = tf.reshape(tf.gather(time, right_idx, axis = 0), [1, streams_length, num_lags, 1])

    X_query = X_left + (time_query[None, :, :, None] - t_left) * (X_right - X_left) / (t_right - t_left)
    
    return X_query



def add_lags_to_streams(X, lags):
    """
    Given an input time streams X adds its lagged versions as extra dimensions,
    where non-integer lagged versions are computed via linear interpolation.
    Inputs
    :X:     an array of time streams of shape (num_paths, streams_length, num_features)
    :lags:  a vector of size (num_lags) containing the lag values 
    Return
    :X_lags: an array of shape (num_paths, streams_length, (num_lags + 1) * num_features)
    """
    # To-do: provide option for interpolation along the first coordinate (if time)
    num_paths, streams_length, num_features = tf.unstack(tf.shape(X))
    num_lags = tf.shape(lags)[0]
    
    time = tf.cast(tf.range(streams_length), settings.float_type) / tf.cast(streams_length, settings.float_type)
    lags = tf.cast(lags, settings.float_type)
    # lags = tf.cast(tf.concat(([0.], lags), axis = 0), settings.float_type)
    time_lags = tf.maximum(time[:,None] - lags[None,:], 0.)
    X_lags = lin_interp_time(time, X, time_lags)
    X_lags = tf.reshape(X_lags, [num_paths, streams_length, -1])
    X_new = tf.concat((X, X_lags), axis = -1)
    return X_new


def extract_unique_and_obs(X, num_features, streams_length):
    """
    Given an input table of streams of shape (num_paths, num_features * streams_length + 1)
    with the first coordinate indicating the length of paths, extracts the actual paths 
    and concatenates them.
    Input
    :X: an [num_paths, num_features * streams_length + 1] array
    Output
    :X_unique:  a size [\\Sum_{i=0}^{num_paths-1} X[i,0], num_features] array
    :X_obs:     an array of size [num_paths * streams_length, num_features] containing all observations with repetitions included
    """
    unique_length = tf.cast(X[:, 0], settings.int_type)
    X = X[:, 1:]
    get_mask_fn =  lambda current_length: tf.concat((tf.ones(current_length, dtype = settings.int_type),
                            tf.zeros([streams_length - current_length], dtype = settings.int_type)), axis = 0)
    mask = tf.reshape(tf.map_fn(get_mask_fn, unique_length), [-1])
    X_obs = tf.reshape(X, [-1, num_features])
    X_unique = tf.boolean_mask(X_obs, mask)
    return X_unique, X_obs
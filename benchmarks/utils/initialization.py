import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import gpflow as gp

def _sample_inducing_tensors(sequences, num_inducing, num_levels, increments):
    Z = []
    sequences_select = sequences[np.random.choice(sequences.shape[0], size=(num_inducing), replace=True)]
    for m in range(1, num_levels+1):        
        if increments:
            obs_idx = [np.random.choice(sequences_select.shape[1]-1, size=(1, m, 1), replace=False) for i in range(num_inducing)]
            obs_idx = np.sort(np.concatenate(obs_idx, axis=0), axis=1)
            obs1_select = np.take_along_axis(sequences_select, obs_idx, axis=1)
            obs2_select = np.take_along_axis(sequences_select, obs_idx + 1, axis=1)
            increments_select = np.concatenate((obs1_select[:, :, None, :], obs2_select[:, :, None, :]), axis=2)
            Z.append(increments_select)
        else:
            obs_idx = [np.random.choice(sequences_select.shape[1], size=(1, m, 1), replace=False) for i in range(num_inducing)]
            obs_idx = np.sort(np.concatenate(obs_idx, axis=0), axis=1)
            obs_select = np.take_along_axis(sequences_select, obs_idx, axis=1)
            Z.append(obs_select)
    Z = np.concatenate(Z, axis=1)
    return Z

def suggest_initial_inducing_tensors(sequences, num_levels, num_inducing, labels=None, increments=False, num_lags=None):
    Z = []
    len_inducing = int(num_levels * (num_levels+1) / 2)
    if labels is not None:
        num_classes = np.unique(labels).size
        bincount = np.bincount(labels)
        # sample from class specific inducing examples
        for c, n_c in enumerate(bincount):
            num_inducing_per_class = int(np.floor(float(n_c) / sequences.shape[0] * num_inducing))
            sequences_class = sequences[labels == c]
            Z.append(_sample_inducing_tensors(sequences_class, num_inducing_per_class, num_levels, increments))
        num_diff = num_inducing - np.sum([z.shape[0] for z in Z])
    else:
        num_diff = num_inducing
    if num_diff > 0:
        Z.append(_sample_inducing_tensors(sequences, num_diff, num_levels, increments))

    Z = np.concatenate(Z, axis=0)
    Z = np.squeeze(Z.reshape([Z.shape[0], len_inducing, -1, Z.shape[-1]]).transpose([1, 0, 2, 3]))
    if num_lags is not None and num_lags > 0:
        if increments:
            Z = np.tile(Z[:, :, :, None, :], (1, 1, 1, num_lags+1, 1)).reshape([Z.shape[0], Z.shape[1], 2, -1])
        else:
            Z = np.tile(Z[:, :, :, None, :], (1, 1, num_lags+1, 1)).reshape([Z.shape[0], Z.shape[1], -1])
    
    Z += 0.4 * np.random.randn(*Z.shape)
    return Z


def _sample_inducing_sequences(sequences, num_inducing, len_inducing):
    Z = []
    sequences_select = sequences[np.random.choice(sequences.shape[0], size=(num_inducing), replace=True)]
    nans_start = np.argmax(np.any(np.isnan(sequences_select), axis=2), axis=1)
    nans_start[nans_start == 0] = sequences.shape[1]
    last_obs_idx = np.concatenate([np.random.choice(range(len_inducing-1, nans_start[i]), size=(1)) for i in range(num_inducing)], axis=0)
    obs_idx = np.stack([last_obs_idx - len_inducing + 1 + i for i in range(len_inducing)], axis=1)[..., None]
    Z = np.take_along_axis(sequences_select, obs_idx, axis=1)
    return Z


def suggest_initial_inducing_sequences(sequences, num_inducing, len_inducing, labels=None):
    Z = []
    if labels is not None:
        num_classes = np.unique(labels).size
        bincount = np.bincount(labels)
        # sample class specific inducing examples
        for c, n_c in enumerate(bincount):
            num_inducing_per_class = int(np.floor(float(n_c) / sequences.shape[0] * num_inducing))
            sequences_class = sequences[labels == c]
            Z.append(_sample_inducing_sequences(sequences_class, num_inducing_per_class, len_inducing))
        num_diff = num_inducing - np.sum([z.shape[0] for z in Z])
    else:
        num_diff = num_inducing

    if num_diff > 0:
        Z.append(_sample_inducing_sequences(sequences, num_diff, len_inducing))
    
    Z = np.concatenate(Z, axis=0)
    
    Z += 0.4 * np.random.randn(*Z.shape)
    return Z


def suggest_initial_lengthscales(X, num_samples=None):
    X = X.reshape([-1, X.shape[-1]])
    X = X[np.logical_not(np.any(np.isnan(X), axis=1))]
    if num_samples is not None and num_samples < X.shape[0]:
        X = X[np.random.choice(X.shape[0], size=(num_samples), replace=False)]
    X = tf.convert_to_tensor(X, gp.settings.float_type)
    l_init = tf.sqrt(tf.reduce_mean(tf.reshape(tf.square(X)[:, None, :] + tf.square(X)[None, :, :] - 2 * X[:, None, :] * X[None, :, :], [-1, tf.shape(X)[1]]), axis=0)
                     * tf.cast(X.shape[1], gp.settings.float_type))
    with tf.Session() as sess:
        l_init = sess.run(l_init)
    return np.maximum(l_init, 1.)
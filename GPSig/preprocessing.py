import numpy as np
# from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class TimeSeriesFeatureScaler(BaseEstimator, TransformerMixin):
    """
    Rescales the features of a time-series via 1 of the 4 modes: 'standard', 'max-abs', 'min-max-plus', 'min-max-sym'.
    Input:
    :mode:
    :percentile: percentile cut-off when computing normalizations.
    """
    
    def __init__(self, mode = 'standard', max_percentile = 99, min_percentile = 1, num_features = None):
        self.mode = mode
        self.max_percentile = max_percentile
        self.min_percentile = min_percentile
        self.num_features = num_features
    
    def _reset(self):
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.shift_
    
    def fit(self, X, y = None):
        if X.ndim < 3 and self.num_features is None:
            raise ValueError('Either specify the parameter num_features or X should have ndim 3.')
        if self.mode is None:
            return self
        self._reset()
        self.num_features = self.num_features if self.num_features is not None else X.shape[2]
        X_samples = X.copy()
        X_samples = X_samples.reshape([-1, self.num_features])
        cut_off_max = np.nanpercentile(X_samples, self.max_percentile, axis=0)
        cut_off_min = np.nanpercentile(X_samples, self.min_percentile, axis=0)
        X_samples[np.where(np.logical_or(X_samples > cut_off_max[None, :], X_samples < cut_off_min[None, :]))] = float('nan')
        if self.mode == 'standard':
            self.shift_ = np.nanmean(X_samples, axis=0)
            self.scale_ = np.nanstd(X_samples, axis=0)
        elif self.mode == 'max-abs':
            self.shift_ = np.zeros((self.num_features))
            self.scale_ = np.nanmax(np.abs(X_samples), axis=0)
        elif self.mode == 'min-max-plus':
            self.shift_ = np.nanmin(X_samples, axis=0)
            self.scale_ = np.nanmax(X_samples, axis=0) - self.shift_
        elif self.mode == 'min-max-sym':
            X_max = np.nanmax(X_samples, axis=0)
            X_min = np.nanmin(X_samples, axis=0)
            self.shift_ = 1.0/2.0 * (X_max + X_min)
            self.scale_ = 1.0/2.0 * (X_max - X_min)
        return self
    
    def transform(self, X, y = None, copy = False):
        check_is_fitted(self, 'scale_')
        n_dims = X.ndim
        if n_dims == 3:
            assert X.shape[2] == self.num_features
        else:
            assert X.shape[1] % self.num_features == 0
        if copy:
            X = X.copy()
        if self.mode is None:
            return X
        X = X.reshape([X.shape[0], -1, self.num_features])
        X = (X - self.shift_[None, None, :]) / self.scale_[None, None, :]
        if n_dims == 2:
            X = X.reshape([X.shape[0], -1])
        return X

def pad_series(max_length, series):
    """
    Takes as input a target length and a multi-dimensional series and pads it with repeating the
    last element until target length is reached.
    Input
    :length: an integer such that stream_length <= max_length
    :series: a series of shape (stream_length, num_features)
    Output
    :series_padded: a series of shape (max_length, num_features) by repeating the last element
    """
    return np.concatenate((series, np.tile(series[-1], [max_length - series.shape[0], 1])), axis = 0)
    

def tabulate_list_of_streams(ts_list, orient_ax = 0, pad_with_na = False):
    """
    Takes as input a list of time-series and constructs a 2D array as list that may be fed as input to machine learning algorithms.
    Since concatenating a path with a constant path does not change its signature, doing this for different length paths is
    a good trick to do without changing the results when fed as input e.g. to the sequentialised kernel.
    Input:
    :ts_list:   A list of time series or sequences, possibly multidimensional and different length
    :orient_ax:     Possible values are 0 and 1. Defaults to 0.
                    if orient_ax = 0, series_list[i] should be a size (stream_length[i], num_features) numpy array
                    if orient_ax = 1, series_list[i] should be a size (num_features, stream_length[i]) numpy array
                    where stream_length[i] is the length of the ith time sequence
    Output:
    :ts_table:  A numpy array of size (len(series_list), max(stream_length)*num_features) with tabulated time series
    """
    # First check whether series are of the right format
    if not np.all([series.ndim == 2 for series in ts_list]):
        raise ValueError('Make sure ndim == 2 for all time-series in the list!')
    
    # Transpose if horizontally oriented
    if orient_ax == 1:
        ts_list = [np.transpose(series) for series in ts_list]
    orient_ax = 0
    feature_ax = 1
    
    num_features = np.asarray([series.shape[feature_ax] for series in ts_list])
    if not np.all(num_features == num_features[0]):
        raise ValueError('Different path dimensions found. Please preprocess time-series beforehand so that all paths contain the same number of features.')
    num_features = num_features[0]
    
    max_length = np.max([series.shape[orient_ax] for series in ts_list])
    
    if not pad_with_na:
        pad_these_series = partial(pad_series, max_length)
    else:
        pad_these_series = lambda x: np.concatenate((x, np.full((max_length - x.shape[0], x.shape[1]), float('nan'))), axis=0)


#    pool = Pool()
    num_sequences = len(ts_list)
    
    ts_list_tabulated = list(tqdm(map(pad_these_series, ts_list), total = num_sequences))
    # ts_table = np.reshape(np.stack(ts_list_tabulated , axis = 0), [num_sequences, max_length * num_features])
    ts_table = np.stack(ts_list_tabulated , axis = 0)
    return ts_table

tabulate_list_of_series = tabulate_list_of_streams

def add_time_to_path(series, num_features = 1):
    """
    Input:
    :series:  A numpy array of size (stream_length * num_features) with possibly repeating elements at the end
    Output:
    :series_with_time:  A numpy array of size (stream_length * (num_features + 1)) with the last time coordinate repeating as many times at the other features.
    """
    # Find the number of repeating elements
    series = np.reshape(series, [-1, num_features])
    length = series.shape[0]
    num_repeating = 1
    while num_repeating < length and np.array_equal(series[- 1 - num_repeating], series[-1]):
        num_repeating += 1
    num_repeating -= 1
    unique_length = series.shape[0] - num_repeating
    time = np.arange(unique_length, dtype=np.float64) / unique_length
    time = np.concatenate((time, np.tile(time[-1], [num_repeating])), axis = 0)
    series = np.concatenate((time[:,None], series), axis = 1)
    return series.flatten()

def add_time_to_table(ts_table, num_features = 1):
    """
    Takes as input a table of tabulated time-series as a size (num_paths, num_features * max_length) numpy array and adds time as an extra coordinate,
    taking into consideration possible repeated elements at the end of each sequence.
    Input:
    :ts_table:      A table of tabulated time-series of size (num_paths, num_features * max_length)
    :num_features:  The number of feature dimensions in each path
    Output:
    :ts_table_with_time:  A table of tabulated time-series of size (num_paths, (num_features + 1) * max_length)
    """
    ndim_format = ts_table.ndim == 3
    if ndim_format:
        ts_table = ts_table.reshape([ts_table.shape[0], -1])
    ts_table_with_time = np.apply_along_axis(lambda series: add_time_to_path(series, num_features), 1, ts_table)
    if ndim_format:
        ts_table_with_time = ts_table_with_time.reshape([ts_table_with_time.shape[0], -1, num_features + 1])
    return ts_table_with_time

def add_time_to_list(ts_list, num_features = 1, len_max = None):
    """
    Takes as input a list of (stream_length_i, num_features) arrays of multivariate, possibly different length time-series,
    and returns a list of (stream_length_i, num_features) array.
    Input:
    :ts_list:       A list of arrays of time-series.
    :num_features:  The number of feature dimensions in each seriees
    :len_max:       None or int, the scaling of time 
    Output:
    :ts_list_with_time:  A list of time-series with (n+1) dimensions
    """
    num_series = len(ts_list)
    if len_max is not None:
        ts_list = [np.concatenate((np.asarray(range(x.shape[0]), dtype=np.float64).reshape([x.shape[0], 1]) / (len_max-1), x), axis=1) for x in ts_list]
    else:
        ts_list = [np.concatenate((np.asarray(range(x.shape[0]), dtype=np.float64).reshape([x.shape[0], 1]) / (x.shape[0]-1), x), axis=1) for x in ts_list]
    return ts_list


        

def add_length_to_path(series, num_features = 1):
    """
    Input:
    :series:  A numpy array of size (stream_length * num_features) with possibly repeating elements at the end
    Output:
    :series_with_time:  A numpy array of size (stream_length * num_features + 1) with an extra non-repeating length counter at the front.
    """
    # Find the number of repeating elements
    series = np.reshape(series, [-1, num_features])
    length = series.shape[0]
    num_repeating = 1
    while num_repeating < length and np.array_equal(series[- 1 - num_repeating], series[-1]):
        num_repeating += 1
    num_repeating -= 1
    unique_length = series.shape[0] - num_repeating
    series = np.concatenate(([unique_length], series.flatten()), axis = 0)
    return series

def add_static_length_coord_to_table(ts_table, num_features = 1):
    """
    Takes as input a table of tabulated time-series as a size (num_paths, num_features * max_length) numpy array and adds
    an extra coordinate at the beginning denoting the length for each series (i.e. total length - repetitions at the end)
    Input:
    :ts_table:      A table of tabulated time-series of size (num_paths, num_features * max_length)
    :num_features:  The number of feature dimensions in each path
    Output:
    :ts_table_with_length:  A table of tabulated time-series of size (num_paths, num_features * max_length + 1)
    """
    ts_table_with_length = np.apply_along_axis(lambda series: add_length_to_path(series, num_features), 1, ts_table)
    return ts_table_with_length

def center_paths(ts_table, num_features = 1, normalize = False):
    """
    Takes as input a table of tabulated time-series as a size (num_paths, num_features * max_length) numpy array centers them.
    Input:
    :ts_table:      A table of tabulated time-series of size (num_paths, num_features * max_length)
    :num_features:  The number of feature dimensions in each path
    Output:
    :ts_table_normalized:  A table of tabulated time-series of size (num_paths, num_features * max_length)
    """
    num_paths = ts_table.shape[0]
    samples = np.zeros((0, num_features))
    for i in range(num_paths):
        x = ts_table[i].reshape([-1, num_features])
        length = x.shape[0]
        num_repeating = 1
        while num_repeating < length and np.array_equal(x[- 1 - num_repeating], x[-1]):
            num_repeating += 1
        num_repeating -= 1
        unique_length = x.shape[0] - num_repeating
        samples = np.concatenate((samples, x[:unique_length]), axis=0)
    num_samples = samples.shape[0]
    samples_mean = samples.mean(axis=0)
    if normalize:
        samples_std = samples.std(axis=0)
        ts_table_normalized = np.reshape((ts_table.reshape([-1, num_features]) - samples_mean[None, :]) / samples_std[None, :], ts_table.shape)
        return ts_table_normalized
    else:
        ts_table_centered = np.reshape(ts_table.reshape([-1, num_features]) - samples_mean[None, :], ts_table.shape)
        return ts_table_centered

def center_samples(ts_table, num_features = 1, normalize = False):
    """
    Takes as input a table of tabulated time-series as a size (num_paths, num_features * max_length) numpy array centers them.
    Input:
    :ts_table:      A table of tabulated time-series of size (num_paths, num_features * max_length)
    :num_features:  The number of feature dimensions in each path
    Output:
    :ts_table_centered:  A table of tabulated time-series of size (num_paths, num_features * max_length)
    """
    num_paths = ts_table.shape[0]
    ts_table_centered = []
    for i in range(num_paths):
        x = ts_table[i].reshape([-1, num_features])
        length = x.shape[0]
        num_repeating = 1
        while num_repeating < length and np.array_equal(x[- 1 - num_repeating], x[-1]):
            num_repeating += 1
        num_repeating -= 1
        unique_length = x.shape[0] - num_repeating
        # print(unique_length)
        sample = x[:unique_length]
        sample_mean = sample.mean(axis=0)
        if normalize:
            sample_std = sample.std(axis=0)
            ts_table_centered.append(((x - sample_mean[None, :]) / sample_std[None, :]).reshape([1, -1]))
        else:
            ts_table_centered.append((x - sample_mean[None, :]).reshape([1, -1]))
    
    ts_table_centered = np.vstack(ts_table_centered)
    return ts_table_centered

def MinMaxScaleFeatures(ts_table, num_features = 1, cut_off = 1):
    """
    Takes as input a table of tabulated time-series as a size (num_paths, num_features * max_length) numpy array and
    normalises each feature dimension to between (-1,1)
    Input:
    :ts_table:      A table of tabulated time-series of size (num_paths, num_features * max_length)
    :num_features:  The number of feature dimensions in each path
    :cut_off:       Percentile cut-off for outliers when computing min/max
    Output:
    :ts_table_scaled:  A table of tabulated time-series of size (num_paths, num_features * max_length)
    """
    ts_concatenated = np.reshape(ts_table, [-1, num_features])
    scale = np.percentile(np.abs(ts_concatenated), percentile, axis = 0).reshape([1, -1])
    ts_concatenated /= scale
    ts_table_normalized = np.reshape(ts_concatenated, ts_table.shape)
    return ts_table_normalized

def normalize_paths(ts_table, num_features = 1, percentile = 97):
    """
    Takes as input a table of tabulated time-series as a size (num_paths, num_features * max_length) numpy array and
    normalises all paths to (0,1) dimension-wise applying the same lengthscales to each path
    Input:
    :ts_table:      A table of tabulated time-series of size (num_paths, num_features * max_length)
    :num_features:  The number of feature dimensions in each path
    Output:
    :ts_table_normalized:  A table of tabulated time-series of size (num_paths, num_features * max_length)
    """
    ts_concatenated = np.reshape(ts_table, [-1, num_features])
    scale = np.percentile(np.abs(ts_concatenated), percentile, axis = 0).reshape([1, -1])
    ts_concatenated /= scale
    ts_table_normalized = np.reshape(ts_concatenated, ts_table.shape)
    return ts_table_normalized
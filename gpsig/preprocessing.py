import numpy as np
from functools import partial
from tqdm import tqdm

def pad_sequence(max_length, pre, seq):
    """
    Takes as input a target length and a multi-dimensional sequence and pads it with repeating the last or first element until target length is reached.
    # Input
    :max_length:    an integer such that len_examples <= max_length
    :sequence:      a sequence of shape (len_examples, num_features)
    # Output
    a sequence of shape (max_length, num_features) by repeating the last element
    """
    if bool(pre):
        return np.concatenate((np.tile(seq[0], [max_length - seq.shape[0], 1]), seq), axis = 0)
    else:
        return np.concatenate((seq, np.tile(seq[-1], [max_length - seq.shape[0], 1])), axis = 0)


def tabulate_list_of_sequences(sequences_list, orient_ax = 0, pad_with = None, pre=False):
    """
    Takes as input a list of sequences and constructs a 2D array as list that may be fed as input to machine learning algorithms.
    Since concatenating a path with a constant path does not change its signature, doing this for different length paths is
    a good trick to do without changing the results when fed as input e.g. to the sequentialised kernel.
    # Input
    :sequences_list:    A list of time sequence or sequences, possibly multidimensional and different length
    :orient_ax:         if orient_ax = 0, sequence_list[i] is (len_examples[i], num_features) 
                        if orient_ax = 1, sequence_list[i] is (num_features, len_examples[i])
    Output:
    :sequences_array:   np array of size (len(sequence_list), max(len_examples), num_features)
    """
    # First check whether sequence are of the right format
    if not np.all([sequence.ndim == 2 for sequence in sequences_list]):
        raise ValueError('Make sure ndim == 2 for all sequences in the list!')
    
    # Transpose if horizontally oriented
    if orient_ax == 1:
        sequences_list = [np.transpose(sequence) for sequence in sequences_list]
    orient_ax = 0
    feature_ax = 1
    
    num_features = np.asarray([sequence.shape[feature_ax] for sequence in sequences_list])
    if not np.all(num_features == num_features[0]):
        raise ValueError('Different path dimensions found. Please preprocess sequences beforehand so that all paths contain the same number of features.')
    num_features = num_features[0]
    
    max_length = np.max([sequence.shape[orient_ax] for sequence in sequences_list])
    
    if pad_with is None:
        pad_these_sequence = partial(pad_sequence, max_length, pre)
    else:
        if pre:
            pad_these_sequence = lambda x: np.concatenate((np.full((max_length - x.shape[0], x.shape[1]), float(pad_with)), x), axis=0)
        else:
            pad_these_sequence = lambda x: np.concatenate((x, np.full((max_length - x.shape[0], x.shape[1]), float(pad_with))), axis=0)

    num_sequences = len(sequences_list)
    
    sequences_list_tabulated = list(tqdm(map(pad_these_sequence, sequences_list), total = num_sequences))
    sequences_array = np.stack(sequences_list_tabulated , axis = 0)
    return sequences_array

def add_time_to_sequence(sequence):
    """
    # Input
    :sequence:              np array of size (len_examples, num_features) with possibly repeating elements at the end, which are detected
    # Output
    :sequence_with_time:    np array of size (len_examples*(num_features + 1)) with the time coordinate repeating as many times at the other features.
    """
    
    length, num_features = sequence.shape
    num_repeating = 1
    while num_repeating < length and np.array_equal(sequence[-1 - num_repeating], sequence[-1]):
        num_repeating += 1
    num_repeating -= 1
    unique_length = sequence.shape[0] - num_repeating
    time = np.arange(unique_length, dtype=np.float64) / (unique_length - 1)
    time = np.concatenate((time, np.tile(time[-1], [num_repeating])), axis = 0)
    sequence = np.concatenate((time[:,None], sequence), axis = 1)
    return sequence.flatten()

def add_time_to_table(sequences_array, num_features = None):
    """
    Takes as input a table of tabulated sequences as a size (num_paths, num_features * max_length) numpy array and adds time as an extra coordinate,
    taking into consideration possible repeated elements at the end of each sequence.
    # Input
    :sequences_array:           A table of tabulated sequences of size (num_paths, num_features * max_length)
    :num_features:              The number of sequence coordinates
    Output:
    :sequences_array_with_time: A table of tabulated sequences of size (num_paths, (num_features + 1) * max_length)
    """
    if sequences_array.ndim == 3:
        if num_features is None:
            num_features = sequences_array.shape[2]
        else:
            assert num_features == sequences_array.shape[2]
    else:
        num_features = num_features or 1
    
    sequences_array = sequences_array.reshape([sequences_array.shape[0], -1, num_features])
    sequences_array_with_time = np.apply_along_axis(lambda sequence: add_time_to_sequence(sequence), 1, sequences_array)
    return sequences_array_with_time

def add_natural_parametrization_to_table(sequences_array, num_features = None):
    """
    Takes as input a table of tabulated sequences as a size (num_paths, num_features * max_length) numpy array and
    adds the natural parametrization as an extra coordinate,
    # Input
    :sequences_array:           A table of tabulated sequences of size (num_paths, num_features * max_length) or (num_paths, num_features, max_length)
    :num_features:              The number of sequence coordinates
    Output:
    :sequences_array_with_time: A table of tabulated sequences of size (num_paths, (num_features + 1) * max_length)
    """
    if sequences_array.ndim == 3:
        if num_features is None:
            num_features = sequences_array.shape[2]
        else:
            assert num_features == sequences_array.shape[2]
    else:
        num_features = num_features or 1
    
    sequences_array = sequences_array.reshape([sequences_array.shape[0], -1, num_features])

    natural_param_array = np.linalg.norm(np.diff(sequences_array, axis=1), axis=2)
    natural_param_array = np.concatenate((np.zeros((sequences_array.shape[0], 1), dtype=np.float64), natural_param_array), axis=1)
    natural_param_array = np.cumsum(natural_param_array, axis=1)

    sequences_array_with_nat_param = np.concatenate((natural_param_array[:, :, None], sequences_array), axis=2)

    return sequences_array_with_nat_param

def add_time_to_list(sequences_list):
    """
    Takes as input a list of (len_examples_i, num_features) arrays of multivariate, possibly different length sequences,
    and returns a list of (len_examples_i, num_features + 1) array.
    # Input
    :sequences_list:            A list of sequences with n coordinates
    # Output
    :sequences_list_with_time:  A list of sequences with (n+1) coordinates
    """
    num_sequences = len(sequences_list)
    sequences_list = [np.concatenate((np.asarray(range(1, x.shape[0]+1), dtype=np.float64).reshape([x.shape[0], 1]) / (x.shape[0]), x), axis=1) for x in sequences_list]
    return sequences_list

def add_natural_parametrization_to_list(sequences_list):
    """
    Takes as input a list of (len_examples_i, num_features) arrays of multivariate, possibly different length sequences,
    and returns a list of (len_examples_i, num_features + 1) array.
    # Input
    :sequences_list:            A list of sequences with n coordinates
    # Output
    :sequences_list_with_time:  A list of sequences with (n+1) coordinates (an extra natural parametrization coordinate)
    """
    num_sequences = len(sequences_list)
    sequences_list = [np.concatenate((np.cumsum(np.concatenate(([0], np.linalg.norm(np.diff(x, axis=0), axis=1)), axis=0), axis=0)[:, None], x), axis=1) for x in sequences_list]
    return sequences_list

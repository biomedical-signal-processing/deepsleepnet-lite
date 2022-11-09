import numpy as np
import random

import itertools

from deepsleeplite.sleep_stages import print_n_samples_each_class

# Extract sequences of length L=seq_length
def get_sequences(x, y, seq_length):
    """
    """
    x_sequences = []
    y_sequences = []

    for sbj in range(len(x)):
        x_sequences_tmp = []
        y_sequences_tmp = []
        for i in range(len(y[sbj]) - (seq_length - 1)):
            x_sequences_tmp.append(x[sbj][i:(seq_length + i)])
            y_sequences_tmp.append(y[sbj][i:(seq_length + i)])
        x_sequences.append(np.asarray(x_sequences_tmp))
        y_sequences.append(np.asarray(y_sequences_tmp))

    return x_sequences, y_sequences

# Balance class sequences - Oversampling
def get_balance_class_sequences_oversample(x, y, seq_length, flipping):

    """
    Balance the number of samples of all classes by (oversampling):
        1. Find the class that has the largest number of samples
        2. Randomly select samples in each class equal to that largest number
    """

    print_n_samples_each_class(y[:, int((seq_length-1)/2)])
    print(" ")

    class_labels = np.unique(y[:, int((seq_length-1)/2)])
    n_max_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y[:, int((seq_length-1)/2)] == c)[0])
        if n_max_classes < n_samples:
            n_max_classes = n_samples

    balance_x = []
    balance_y = []
    for c in class_labels:

        idx = np.where(y[:, int((seq_length-1)/2)] == c)[0]
        n_samples = len(idx)
        n_repeats = int(n_max_classes / n_samples)

        if flipping:

            # Oversampling with flipping
            x_augmented = np.concatenate((x[idx], np.negative(x[idx])))
            y_augmented = np.concatenate((y[idx], y[idx]))
            indices = np.arange(len(x_augmented))
            np.random.shuffle(indices)
            augmented_idx = random.choices(indices, k=len(idx))

            tmp_x = np.vstack((x[idx], np.repeat(x_augmented[augmented_idx], n_repeats-1, axis=0)))
            tmp_y = np.vstack((y[idx], np.repeat(y_augmented[augmented_idx], n_repeats-1, axis=0)))

            n_remains = n_max_classes - len(tmp_x)
            if n_remains > 0:
                augmented_sub_idx = random.choices(indices, k=n_remains)
                tmp_x = np.vstack([tmp_x, x_augmented[augmented_sub_idx]])
                tmp_y = np.vstack([tmp_y, y_augmented[augmented_sub_idx]])


        else:

            tmp_x = np.repeat(x[idx], n_repeats, axis=0)
            tmp_y = np.repeat(y[idx], n_repeats, axis=0)

            n_remains = n_max_classes - len(tmp_x)
            if n_remains > 0:
                sub_idx = np.random.permutation(idx)[:n_remains]
                tmp_x = np.vstack([tmp_x, x[sub_idx]])
                tmp_y = np.vstack([tmp_y, y[sub_idx]])


        balance_x.append(tmp_x)
        balance_y.append(tmp_y)
    balance_x = np.vstack(balance_x)
    balance_y = np.vstack(balance_y)

    print_n_samples_each_class(balance_y[:, int((seq_length - 1) / 2)])
    print(" ")

    return balance_x, balance_y


def iterate_minibatches_train(inputs, targets, batch_size, seq_length, shuffle=False):
    """
    Generate a generator that return a batch of inputs and targets.
    """
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)

    count = range(0, len(inputs), batch_size)
    for start_idx in count:
        if count[-1] == start_idx:
            indx1 = [random.randint(0, start_idx) for p in range(0, batch_size-(len(inputs)-start_idx))]
            excerpt1 = indices[indx1]
            excerpt2 = indices[start_idx:]
            excerpt = np.concatenate((excerpt1, excerpt2), axis=0)
        else:
            excerpt = indices[start_idx:start_idx + batch_size]

        yield inputs[excerpt], targets[excerpt][:, int((seq_length-1)/2)], targets[excerpt]

def iterate_minibatches_valid_test(inputs, targets, batch_size, seq_length, shuffle=False):
    """
    Generate a generator that return a batch of inputs and targets.
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)

        yield inputs[excerpt], targets[excerpt][:, int((seq_length-1)/2)], targets[excerpt]


def iterate_minibatches_prediction(inputs, targets, batch_size, seq_length):
    assert len(inputs) == len(targets)
    n_inputs = len(inputs)
    batch_len = n_inputs // batch_size

    epochs_shifts = batch_len - seq_length

    seq_inputs = np.zeros((batch_size, batch_len) + inputs.shape[1:], dtype=inputs.dtype)
    seq_targets = np.zeros((batch_size, batch_len) + targets.shape[1:], dtype=targets.dtype)

    for i in range(batch_size):
        seq_inputs[i] = inputs[i * batch_len:(i + 1) * batch_len]
        seq_targets[i] = targets[i * batch_len:(i + 1) * batch_len]

    for i in range(epochs_shifts + 1):
        x = seq_inputs[:, i: seq_length + i]
        y = seq_targets[:, int((seq_length-1)/2) + i]
        flatten_x = x.reshape((batch_size, seq_length*inputs.shape[1]) + inputs.shape[2:])
        flatten_y = y.reshape((-1,) + targets.shape[1:])

        #tmp change
        flatten_y_seq = seq_targets[:, i: seq_length + i].reshape((-1,) + targets.shape[1:])

        yield flatten_x, flatten_y, flatten_y_seq, batch_len, epochs_shifts

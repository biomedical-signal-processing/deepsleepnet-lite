import tensorflow as tf
import numpy as np

from tensorflow.python.training import moving_averages

def _create_variable(name, shape, initializer, freeze=False):
    var = tf.compat.v1.get_variable(name, shape,  dtype=tf.dtypes.float32, initializer=initializer, trainable=not freeze)
    return var

def variable_with_weight_decay(name, shape, wd=None, freeze=False, normal_initializer=True):
    # Get the number of input and output parameters
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4:
        receptive_field_size = np.prod(shape[:2])
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    else:
        # no specific assumptions
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))

    # He et al. 2015 - http://arxiv.org/abs/1502.01852
    stddev = np.sqrt(2.0 / fan_in)
    if normal_initializer:
        initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.dtypes.float32)
    else:
        initializer = tf.random_uniform_initializer(minval=-1, maxval=0, seed=None)

    # # Xavier
    # initializer = tf.contrib.layers.xavier_initializer()

    # Create or get the existing variable
    var = _create_variable(
        name,
        shape,
        initializer,
        freeze=freeze
    )

    # L2 weight decay
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="weight_loss")  # tf.mul
        tf.compat.v1.add_to_collection("losses", weight_decay)

    return var

def conv_1d(name, input_var, filter_shape, stride, padding="SAME",
            bias=None, wd=None):
    with tf.compat.v1.variable_scope(name) as scope:
        # Trainable parameters
        kernel = variable_with_weight_decay(
            "weights",
            shape=filter_shape,
            wd=wd
        )

        # Convolution
        output_var = tf.nn.conv2d(
            input_var,
            kernel,
            [1, stride, 1, 1],
            padding=padding
        )

        # Bias
        if bias is not None:
            biases = _create_variable(
                "biases",
                [filter_shape[-1]],
                tf.constant_initializer(bias, dtype=tf.dtypes.float32)
            )
            output_var = tf.nn.bias_add(output_var, biases)

        return output_var, kernel

def max_pool_1d(name, input_var, pool_size, stride, padding="SAME"):
    output_var = tf.nn.max_pool2d(
        input_var,
        ksize=[1, pool_size, 1, 1],
        strides=[1, stride, 1, 1],
        padding=padding,
        name=name
    )

    return output_var

def batch_norm(name, input_var, is_train, decay=0.999, epsilon=1e-5):
    """Batch normalization modified from BatchNormLayer in Tensorlayer.
    Source: <https://github.com/zsdonghao/tensorlayer/blob/master/tensorlayer/layers.py#L2190>
    """

    inputs_shape = input_var.get_shape()
    axis = list(range(len(inputs_shape) - 1))
    params_shape = inputs_shape[-1:]

    with tf.compat.v1.variable_scope(name) as scope:
        # Trainable beta and gamma variables
        beta = tf.compat.v1.get_variable('beta',
                               shape=params_shape, dtype=tf.dtypes.float32,
                               initializer=tf.zeros_initializer(dtype=tf.dtypes.float32))
        gamma = tf.compat.v1.get_variable('gamma',
                                shape=params_shape, dtype=tf.dtypes.float32,
                                initializer=tf.random_normal_initializer(mean=1.0, stddev=0.002, dtype=tf.dtypes.float32))

        # Moving mean and variance updated during training
        moving_mean = tf.compat.v1.get_variable('moving_mean',
                                      params_shape, dtype=tf.dtypes.float32,
                                      initializer=tf.zeros_initializer(dtype=tf.dtypes.float32),
                                      trainable=False)
        moving_variance = tf.compat.v1.get_variable('moving_variance',
                                          params_shape, dtype=tf.dtypes.float32,
                                          initializer=tf.constant_initializer(1., dtype=tf.dtypes.float32),
                                          trainable=False)

        # Compute mean and variance along axis
        batch_mean, batch_variance = tf.nn.moments(input_var, axis, name='moments')

        # Define ops to update moving_mean and moving_variance
        update_moving_mean = moving_averages.assign_moving_average(moving_mean, batch_mean, decay, zero_debias=False)
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, batch_variance, decay,
                                                                       zero_debias=False)

        # Define a function that :
        # 1. Update moving_mean & moving_variance with batch_mean & batch_variance
        # 2. Then return the batch_mean & batch_variance
        def mean_var_with_update():
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                return tf.identity(batch_mean), tf.identity(batch_variance)

        # Perform different ops for training and testing
        if is_train:
            mean, variance = mean_var_with_update()
            normed = tf.nn.batch_normalization(input_var, mean, variance, beta, gamma, epsilon)

        else:
            normed = tf.nn.batch_normalization(input_var, moving_mean, moving_variance, beta, gamma, epsilon)

        return normed


def flatten(name, input_var):
    dim = 1
    for d in input_var.get_shape()[1:].as_list():
        dim *= d
    output_var = tf.reshape(input_var,
                            shape=[-1, dim],
                            name=name)

    return output_var

def fc(name, input_var, n_hiddens, bias=None, wd=None, freeze_layer=False, normal_initializer=True):
    with tf.compat.v1.variable_scope(name) as scope:
        # Get input dimension
        input_dim = input_var.get_shape()[-1].value

        # Trainable parameters
        weights = variable_with_weight_decay(
            "weights",
            shape=[input_dim, n_hiddens],
            wd=wd,
            freeze=freeze_layer, normal_initializer=normal_initializer
        )

        # Multiply weights
        output_var = tf.matmul(input_var, weights)

        # Bias
        if bias is not None:
            biases = _create_variable(
                "biases",
                [n_hiddens],
                tf.constant_initializer(bias, dtype=tf.dtypes.float32),
                freeze=freeze_layer
            )
            output_var = tf.add(output_var, biases)

        return output_var

def label_smoothing_uniform(name, target_var, K, alfa):
    with tf.compat.v1.variable_scope(name) as scope:
        target_var_one_hot = tf.one_hot(target_var, depth=5)
        target_var_smoothed = alfa * (1 / K) + (1 - alfa) * target_var_one_hot
        return target_var_smoothed

def label_smoothing_statistics(name, target_var, target_statistics, alfa):
    with tf.compat.v1.variable_scope(name) as scope:
        target_var_one_hot = tf.one_hot(target_var, depth=5)
        target_var_smoothed = alfa * target_statistics + (1 - alfa) * target_var_one_hot
        return target_var_smoothed


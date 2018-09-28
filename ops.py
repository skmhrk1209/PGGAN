from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def spectral_normalization(input):
    ''' spectral normalization

        [Spectral Normalization for Generative Adversarial Networks]
        (https://arxiv.org/pdf/1802.05957.pdf)

        this implementation is from google 
        (https://github.com/google/compare_gan/blob/master/compare_gan/src/gans/ops.py)
    '''

    if len(input.shape) < 2:
        raise ValueError("Spectral norm can only be applied to multi-dimensional tensors")

    # The paper says to flatten convnet kernel weights from (C_out, C_in, KH, KW)
    # to (C_out, C_in * KH * KW). But Sonnet's and Compare_gan's Conv2D kernel
    # weight shape is (KH, KW, C_in, C_out), so it should be reshaped to
    # (KH * KW * C_in, C_out), and similarly for other layers that put output
    # channels as last dimension.
    # n.b. this means that w here is equivalent to w.T in the paper.
    w = tf.reshape(input, [-1, input.shape[-1]])

    # Persisted approximation of first left singular vector of matrix `w`.

    u_var = tf.get_variable(
        name=input.name.replace(":", "") + "/u_var",
        shape=[w.shape[0], 1],
        dtype=w.dtype,
        initializer=tf.random_normal_initializer(),
        trainable=False
    )
    u = u_var

    # Use power iteration method to approximate spectral norm.
    # The authors suggest that "one round of power iteration was sufficient in the
    # actual experiment to achieve satisfactory performance". According to
    # observation, the spectral norm become very accurate after ~20 steps.

    power_iteration_rounds = 1
    for _ in range(power_iteration_rounds):
        # `v` approximates the first right singular vector of matrix `w`.
        v = tf.nn.l2_normalize(tf.matmul(tf.transpose(w), u), dim=None, epsilon=1e-12)
        u = tf.nn.l2_normalize(tf.matmul(w, v), dim=None, epsilon=1e-12)

    # Update persisted approximation.
    with tf.control_dependencies([tf.assign(u_var, u, name="update_u")]):
        u = tf.identity(u)

    # The authors of SN-GAN chose to stop gradient propagating through u and v.
    # In johnme@'s experiments it wasn't clear that this helps, but it doesn't
    # seem to hinder either so it's kept in order to be a faithful implementation.
    u = tf.stop_gradient(u)
    v = tf.stop_gradient(v)

    # Largest singular value of `w`.
    norm_value = tf.matmul(tf.matmul(tf.transpose(u), w), v)
    norm_value.shape.assert_is_fully_defined()
    norm_value.shape.assert_is_compatible_with([1, 1])

    w_normalized = w / norm_value

    # Unflatten normalized weights to match the unnormalized tensor.
    w_tensor_normalized = tf.reshape(w_normalized, input.shape)
    return w_tensor_normalized


def dense(inputs, units, name="dense", reuse=None, apply_spectral_normalization=False):
    ''' linear layer for spectral normalization
        for weight normalization, use variable instead of tf.layers.dense
    '''

    with tf.variable_scope(name, reuse=reuse):

        weight = tf.get_variable(
            name="weight",
            shape=[inputs.shape[1], units],
            dtype=tf.float32,
            initializer=tf.variance_scaling_initializer(),
            trainable=True
        )

        if apply_spectral_normalization:

            weight = spectral_normalization(weight)

        bias = tf.get_variable(
            name="bias",
            shape=[units],
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            trainable=True
        )

        inputs = tf.matmul(inputs, weight) + bias

        return inputs


def conv2d(inputs, filters, kernel_size, strides, data_format,
           name="conv2d", reuse=None, apply_spectral_normalization=False):
    ''' convolution layer for spectral normalization
        for weight normalization, use variable instead of tf.layers.conv2d
    '''

    with tf.variable_scope(name, reuse=reuse):

        data_format_abbr = "NCHW" if data_format == "channels_first" else "NHWC"

        in_filters = inputs.shape[1] if data_format_abbr == "NCHW" else inputs.shape[3]

        kernel = tf.get_variable(
            name="kernel",
            shape=kernel_size + [in_filters, filters],
            dtype=tf.float32,
            initializer=tf.variance_scaling_initializer(),
            trainable=True
        )

        if apply_spectral_normalization:

            kernel = spectral_normalization(kernel)

        strides = [1] + [1] + strides if data_format_abbr == "NCHW" else [1] + strides + [1]

        inputs = tf.nn.conv2d(
            input=inputs,
            filter=kernel,
            strides=strides,
            padding="SAME",
            data_format=data_format_abbr
        )

        bias = tf.get_variable(
            name="bias",
            shape=[filters],
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            trainable=True
        )

        inputs = tf.nn.bias_add(
            value=inputs,
            bias=bias,
            data_format=data_format_abbr
        )

        return inputs


def deconv2d(inputs, filters, kernel_size, strides, data_format,
             name="deconv2d", reuse=None, apply_spectral_normalization=False):
    ''' deconvolution layer for spectral normalization
        for weight normalization, use variable instead of tf.layers.conv2d_transpose
    '''

    with tf.variable_scope(name, reuse=reuse):

        data_format_abbr = "NCHW" if data_format == "channels_first" else "NHWC"

        in_filters = inputs.shape[1] if data_format_abbr == "NCHW" else inputs.shape[3]

        kernel = tf.get_variable(
            name="kernel",
            shape=kernel_size + [filters, in_filters],
            dtype=tf.float32,
            initializer=tf.variance_scaling_initializer(),
            trainable=True
        )

        if apply_spectral_normalization:

            kernel = spectral_normalization(kernel)

        strides = [1] + [1] + strides if data_format_abbr == "NCHW" else [1] + strides + [1]

        output_shape = tf.shape(inputs) * strides
        output_shape = (tf.concat([output_shape[0:1], [filters], output_shape[2:4]], axis=0) if data_format_abbr == "NCHW" else
                        tf.concat([output_shape[0:1], output_shape[1:3], [filters]], axis=0))

        inputs = tf.nn.conv2d_transpose(
            value=inputs,
            filter=kernel,
            output_shape=output_shape,
            strides=strides,
            padding="SAME",
            data_format=data_format_abbr
        )

        bias = tf.get_variable(
            name="bias",
            shape=[filters],
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            trainable=True
        )

        inputs = tf.nn.bias_add(
            value=inputs,
            bias=bias,
            data_format=data_format_abbr
        )

        return inputs


def residual_block(inputs, filters, strides, normalization, activation, data_format, training,
                   name="residual_block", reuse=None, apply_spectral_normalization=False):
    ''' preactivation building residual block for spectral normalization

        normalization then activation then convolution as described by: 
        [Identity Mappings in Deep Residual Networks]
        (https://arxiv.org/pdf/1603.05027.pdf)
    '''

    with tf.variable_scope(name, reuse=reuse):

        if normalization:

            inputs = normalization(inputs, data_format, training)

        if activation:

            inputs = activation(inputs)

        shortcut = conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=[1, 1],
            strides=strides,
            data_format=data_format,
            name="conv2d_0",
            apply_spectral_normalization=apply_spectral_normalization
        )

        inputs = conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=[3, 3],
            strides=strides,
            data_format=data_format,
            name="conv2d_1",
            apply_spectral_normalization=apply_spectral_normalization
        )

        if normalization:

            inputs = normalization(inputs, data_format, training)

        if activation:

            inputs = activation(inputs)

        inputs = conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=[3, 3],
            strides=[1, 1],
            data_format=data_format,
            name="conv2d_2",
            apply_spectral_normalization=apply_spectral_normalization
        )

        inputs += shortcut

        return inputs


def unpooling2d(inputs, pool_size, data_format):
    ''' upsampling operation with zero padding

        [The GAN Landscape: Losses, Architectures, Regularization, and Normalization]
        (https://arxiv.org/pdf/1807.04720.pdf)

        authors used unpool function from github
        (https://github.com/tensorflow/tensorflow/issues/2169)

        my implementation is complicated but more generic and faster
    '''

    if data_format == "channels_last":
        inputs = tf.transpose(inputs, perm=[0, 3, 1, 2])

    inputs = tf.reshape(inputs, shape=[-1, inputs.shape[0], inputs.shape[1], inputs.shape[2] * inputs.shape[3], 1])

    paddings = [[0, 0], [0, 0], [0, 0], [0, pool_size[1] - 1]]
    inputs = tf.pad(inputs, paddings=paddings, mode="CONSTANT", constant_values=0)

    inputs = tf.reshape(inputs, shape=[-1, inputs.shape[1], inputs.shape[2], inputs.shape[3] * pool_size[1]])

    paddings = [[0, 0], [0, 0], [0, 0], [0, inputs.shape[3] * pool_size[1] * (pool_size[0] - 1)]]
    inputs = tf.pad(inputs, paddings=paddings, mode="CONSTANT", constant_values=0)

    inputs = tf.reshape(inputs, shape=[-1, inputs.shape[1], inputs.shape[2] * pool_size[0], inputs.shape[3] * pool_size[1]])

    if data_format == "channels_last":
        inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])

    return inputs


def global_average_pooling2d(inputs, data_format):

    return tf.reduce_mean(
        input_tensor=inputs,
        axis=[2, 3] if data_format == "channels_first" else [1, 2]
    )


def layer_normalization(inputs, data_format, training):

    return tf.contrib.layers.layer_norm(
        inputs=inputs,
        center=True,
        scale=True,
        trainable=True
    )


def instance_normalization(inputs, data_format, training):

    return tf.contrib.layers.instance_norm(
        inputs=inputs,
        center=True,
        scale=True,
        trainable=True,
        data_format="NCHW" if data_format == "channels_first" else "NHWC"
    )


def batch_normalization(inputs, data_format, training):

    return tf.contrib.layers.batch_norm(
        inputs=inputs,
        center=True,
        scale=True,
        is_training=training,
        trainable=True,
        data_format="NCHW" if data_format == "channels_first" else "NHWC"
    )

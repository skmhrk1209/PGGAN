from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import collections
from . import ops


def lerp(a, b, t):
    return a + (b - a) * t


def log2(m, n):
    x = 0
    while (m << x) < n:
        x += 1
    return x


def fixed_conv2d(inputs, in_filters, out_filters, kernel_size, strides, data_format,
                 name="conv2d", reuse=None, apply_spectral_normalization=False):
    ''' convolution layer for spectral normalization
        for weight normalization, use variable instead of tf.layers.conv2d
    '''

    with tf.variable_scope(name, reuse=reuse):

        data_format_abbr = "NCHW" if data_format == "channels_first" else "NHWC"

        kernel = tf.get_variable(
            name="kernel",
            shape=kernel_size + [in_filters, out_filters],
            dtype=tf.float32,
            initializer=tf.variance_scaling_initializer(),
            trainable=True
        )

        if apply_spectral_normalization:

            kernel = ops.spectral_normalization(
                input=kernel,
                name="spectral_normalization"
            )

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
            shape=[out_filters],
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


def fixed_deconv2d(inputs, out_filters, in_filters, kernel_size, strides, data_format,
                   name="deconv2d", reuse=None, apply_spectral_normalization=False):
    ''' deconvolution layer for spectral normalization
        for weight normalization, use variable instead of tf.layers.conv2d_transpose
    '''

    with tf.variable_scope(name, reuse=reuse):

        data_format_abbr = "NCHW" if data_format == "channels_first" else "NHWC"

        kernel = tf.get_variable(
            name="kernel",
            shape=kernel_size + [out_filters, in_filters],
            dtype=tf.float32,
            initializer=tf.variance_scaling_initializer(),
            trainable=True
        )

        if apply_spectral_normalization:

            kernel = ops.spectral_normalization(
                input=kernel,
                name="spectral_normalization"
            )

        strides = [1] + [1] + strides if data_format_abbr == "NCHW" else [1] + strides + [1]

        output_shape = tf.shape(inputs) * strides
        output_shape = (tf.concat([output_shape[0:1], [out_filters], output_shape[2:4]], axis=0) if data_format_abbr == "NCHW" else
                        tf.concat([output_shape[0:1], output_shape[1:3], [out_filters]], axis=0))

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
            shape=[out_filters],
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


class Generator(object):
    '''
    Progressive Growing GAN Architecture
    [Progressive Growing of GANs for Improved Quality, Stability, and Variation]
    (https://arxiv.org/pdf/1710.10196.pdf)

    based on SNDCGAN
    [Spectral Normalization for Generative Adversarial Networks]
    (https://arxiv.org/pdf/1802.05957.pdf)
    '''

    def __init__(self, min_resolution, max_resolution, max_filters, data_format):

        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.max_filters = max_filters
        self.data_format = data_format

    def __call__(self, inputs, coloring_index, training, name="generator", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            def grow(inputs, index):
                '''
                    very complicated but efficient architecture
                    each layer has two output paths: feature_maps and images
                    whether which path is evaluated at runtime
                    depends on variable "coloring_index"
                    but, all possible pathes must be constructed at compile time
                '''

                with tf.variable_scope("layer_{}".format(index)):

                    ceiled_coloring_index = tf.cast(tf.ceil(coloring_index), tf.int32)

                    if index == 0:

                        feature_maps = self.dense_block(
                            inputs=inputs,
                            index=index,
                            training=training
                        )

                        return feature_maps

                    elif index == 1:

                        feature_maps = grow(inputs, index - 1)

                        images = self.color_block(
                            inputs=feature_maps,
                            index=index,
                            training=training
                        )

                        feature_maps = self.deconv2d_block(
                            inputs=feature_maps,
                            index=index,
                            training=training
                        )

                        return feature_maps, images

                    elif index == log2(self.min_resolution, self.max_resolution) + 1:

                        feature_maps, images = grow(inputs, index - 1)

                        old_images = ops.upsampling2d(
                            inputs=images,
                            factors=[2, 2],
                            data_format=self.data_format
                        )

                        new_images = self.color_block(
                            inputs=feature_maps,
                            index=index,
                            training=training
                        )

                        images = tf.case(
                            pred_fn_pairs={
                                tf.greater(index, ceiled_coloring_index): lambda: old_images,
                                tf.less(index, ceiled_coloring_index): lambda: new_images
                            },
                            default=lambda: lerp(
                                a=old_images,
                                b=new_images,
                                t=coloring_index - (index - 1)
                            ),
                            exclusive=True
                        )

                        return images

                    else:

                        feature_maps, images = grow(inputs, index - 1)

                        old_images = ops.upsampling2d(
                            inputs=images,
                            factors=[2, 2],
                            data_format=self.data_format
                        )

                        new_images = self.color_block(
                            inputs=feature_maps,
                            index=index,
                            training=training
                        )

                        images = tf.case(
                            pred_fn_pairs={
                                tf.greater(index, ceiled_coloring_index): lambda: old_images,
                                tf.less(index, ceiled_coloring_index): lambda: new_images
                            },
                            default=lambda: lerp(
                                a=old_images,
                                b=new_images,
                                t=coloring_index - (index - 1)
                            ),
                            exclusive=True
                        )

                        feature_maps = self.deconv2d_block(
                            inputs=feature_maps,
                            index=index,
                            training=training
                        )

                        return feature_maps, images

            return grow(inputs, log2(self.min_resolution, self.max_resolution) + 1)

    def dense_block(self, inputs, index, training, name="dense_block", reuse=None):

        with tf.variable_scope("dense_block", reuse=None):

            resolution = self.min_resolution << index
            filters = self.max_filters >> index

            inputs = ops.dense(
                inputs=inputs,
                units=resolution * resolution * filters
            )

            inputs = ops.batch_normalization(
                inputs=inputs,
                data_format=self.data_format,
                training=training
            )

            inputs = tf.nn.relu(inputs)

            inputs = tf.reshape(
                tensor=inputs,
                shape=[-1, resolution, resolution, filters]
            )

            if self.data_format == "channels_first":

                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            return inputs

    def deconv2d_block(self, inputs, index, training, name="deconv2d_block", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            inputs = fixed_deconv2d(
                inputs=inputs,
                out_filters=self.max_filters >> index,
                in_filters=self.max_filters >> (index - 1),
                kernel_size=[4, 4],
                strides=[2, 2],
                data_format=self.data_format
            )

            inputs = ops.batch_normalization(
                inputs=inputs,
                data_format=self.data_format,
                training=training
            )

            inputs = tf.nn.relu(inputs)

            return inputs

    def color_block(self, inputs, index, training, name="color_block", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            inputs = fixed_deconv2d(
                inputs=inputs,
                out_filters=3,
                in_filters=self.max_filters >> (index - 1),
                kernel_size=[3, 3],
                strides=[1, 1],
                data_format=self.data_format
            )

            inputs = tf.nn.sigmoid(inputs)

            return inputs


class Discriminator(object):
    '''
    Progressive Growing GAN Architecture
    [Progressive Growing of GANs for Improved Quality, Stability, and Variation]
    (https://arxiv.org/pdf/1710.10196.pdf)

    based on SNDCGAN
    [Spectral Normalization for Generative Adversarial Networks]
    (https://arxiv.org/pdf/1802.05957.pdf)
    '''

    def __init__(self, min_resolution, max_resolution, max_filters, data_format):

        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.max_filters = max_filters
        self.data_format = data_format

    def __call__(self, inputs, coloring_index, training, name="discriminator", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            def grow(feature_maps, images, index):
                '''
                    very complicated but efficient architecture
                    each layer has two output paths: feature_maps and images
                    whether which path is evaluated at runtime
                    depends on variable "coloring_index"
                    but, all possible pathes must be constructed at compile time
                '''

                with tf.variable_scope("layer_{}".format(index)):

                    floored_coloring_index = tf.cast(tf.floor(coloring_index), tf.int32)

                    if index == 0:

                        logits = self.dense_block(
                            inputs=feature_maps,
                            index=index,
                            training=training
                        )

                        return logits

                    elif index == 1:

                        old_feature_maps = self.color_block(
                            inputs=images,
                            index=index,
                            training=training
                        )

                        new_feature_maps = self.conv2d_block(
                            inputs=feature_maps,
                            index=index,
                            training=training
                        )

                        feature_maps = tf.case(
                            pred_fn_pairs={
                                tf.greater(index, floored_coloring_index): lambda: old_feature_maps,
                                tf.less(index, floored_coloring_index): lambda: new_feature_maps
                            },
                            default=lambda: lerp(
                                a=old_feature_maps,
                                b=new_feature_maps,
                                t=coloring_index - index
                            ),
                            exclusive=True
                        )

                        return grow(feature_maps, None, index - 1)

                    elif index == log2(self.min_resolution, self.max_resolution) + 1:

                        feature_maps = self.color_block(
                            inputs=images,
                            index=index,
                            training=training
                        )

                        images = ops.downsampling2d(
                            inputs=images,
                            factors=[2, 2],
                            data_format=self.data_format
                        )

                        return grow(feature_maps, images, index - 1)

                    else:

                        old_feature_maps = self.color_block(
                            inputs=images,
                            index=index,
                            training=training
                        )

                        new_feature_maps = self.conv2d_block(
                            inputs=feature_maps,
                            index=index,
                            training=training
                        )

                        feature_maps = tf.case(
                            pred_fn_pairs={
                                tf.greater(index, floored_coloring_index): lambda: old_feature_maps,
                                tf.less(index, floored_coloring_index): lambda: new_feature_maps
                            },
                            default=lambda: lerp(
                                a=old_feature_maps,
                                b=new_feature_maps,
                                t=coloring_index - index
                            ),
                            exclusive=True
                        )

                        images = ops.downsampling2d(
                            inputs=images,
                            factors=[2, 2],
                            data_format=self.data_format
                        )

                        return grow(feature_maps, images, index - 1)

            return grow(None, inputs, log2(self.min_resolution, self.max_resolution) + 1)

    def dense_block(self, inputs, index, training, name="dense_block", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            inputs = tf.layers.flatten(inputs)

            inputs = ops.dense(
                inputs=inputs,
                units=1,
                apply_spectral_normalization=True
            )

            return inputs

    def conv2d_block(self, inputs, index, training, name="conv2d_block", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            inputs = fixed_conv2d(
                inputs=inputs,
                in_filters=self.max_filters >> index,
                out_filters=self.max_filters >> (index - 1),
                kernel_size=[4, 4],
                strides=[2, 2],
                data_format=self.data_format,
                apply_spectral_normalization=True
            )

            inputs = tf.nn.leaky_relu(inputs)

            return inputs

    def color_block(self, inputs, index, training, name="color_block", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            inputs = fixed_conv2d(
                inputs=inputs,
                in_filters=3,
                out_filters=self.max_filters >> (index - 1),
                kernel_size=[3, 3],
                strides=[1, 1],
                data_format=self.data_format,
                apply_spectral_normalization=True
            )

            inputs = tf.nn.leaky_relu(inputs)

            return inputs

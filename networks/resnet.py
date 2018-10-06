#=================================================================================================#
# Progressive Growing GAN Architecture
#
# [Progressive Growing of GANs for Improved Quality, Stability, and Variation]
# (https://arxiv.org/pdf/1710.10196.pdf)
#
# based on ResNet
# [Deep Residual Learning for Image Recognition]
# (https://arxiv.org/pdf/1512.03385.pdf)
#=================================================================================================#

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


class Generator(object):

    def __init__(self, min_resolution, max_resolution, min_filters, max_filters, data_format):

        if log2(min_resolution, max_resolution) != log2(min_filters, max_filters):
            raise ValueError("Invalid number of filters")

        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.min_filters = min_filters
        self.max_filters = max_filters
        self.data_format = data_format

    def __call__(self, inputs, coloring_index, training, name="generator", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            #========================================================================#
            # very complicated but efficient architecture
            # each layer has two output paths: feature_maps and images
            # whether which path is evaluated at runtime
            # depends on variable "coloring_index"
            # but, all possible pathes must be constructed at compile time
            #========================================================================#
            def grow(inputs, index):

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
                            data_format=self.data_format,
                            dynamic=False
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
                            data_format=self.data_format,
                            dynamic=False
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

        with tf.variable_scope(name, reuse=None):

            resolution = self.min_resolution << index
            filters = self.max_filters >> index

            inputs = ops.dense(
                inputs=inputs,
                units=resolution * resolution * filters,
                name="dense_0"
            )

            inputs = tf.reshape(
                tensor=inputs,
                shape=[-1, resolution, resolution, filters]
            )

            if self.data_format == "channels_first":

                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            return inputs

    def deconv2d_block(self, inputs, index, training, name="deconv2d_block", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            inputs = ops.upsampling2d(
                inputs=inputs,
                factors=[2, 2],
                data_format=self.data_format,
                dynamic=False
            )

            inputs = ops.residual_block(
                inputs=inputs,
                filters=self.max_filters >> index,
                strides=[1, 1],
                normalization=ops.batch_normalization,
                activation=tf.nn.relu,
                data_format=self.data_format,
                training=training,
                name="residual_block_0"
            )

            return inputs

    def color_block(self, inputs, index, training, name="color_block", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            inputs = ops.batch_normalization(
                inputs=inputs,
                data_format=self.data_format,
                training=training,
                name="batch_normalization_0"
            )

            inputs = tf.nn.relu(inputs)

            inputs = ops.conv2d(
                inputs=inputs,
                filters=3,
                kernel_size=[3, 3],
                strides=[1, 1],
                data_format=self.data_format,
                name="conv2d_0"
            )

            inputs = tf.nn.sigmoid(inputs)

            return inputs


class Discriminator(object):

    def __init__(self, min_resolution, max_resolution, min_filters, max_filters, data_format):

        if log2(min_resolution, max_resolution) != log2(min_filters, max_filters):
            raise ValueError("Invalid number of filters")

        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.min_filters = min_filters
        self.max_filters = max_filters
        self.data_format = data_format

    def __call__(self, inputs, coloring_index, training, name="discriminator", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            #========================================================================#
            # very complicated but efficient architecture
            # each layer has two output paths: feature_maps and images
            # whether which path is evaluated at runtime
            # depends on variable "coloring_index"
            # but, all possible pathes must be constructed at compile time
            #========================================================================#
            def grow(feature_maps, images, index):

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

            inputs = tf.nn.relu(inputs)

            inputs = ops.global_average_pooling2d(
                inputs=inputs,
                data_format=self.data_format
            )

            inputs = ops.dense(
                inputs=inputs,
                units=1,
                apply_spectral_normalization=True,
                name="dense_0"
            )

            return inputs

    def conv2d_block(self, inputs, index, training, name="conv2d_block", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            inputs = ops.residual_block(
                inputs=inputs,
                filters=self.max_filters >> (index - 1),
                strides=[1, 1],
                data_format=self.data_format,
                apply_spectral_normalization=True,
                normalization=None,
                training=training,
                activation=tf.nn.relu,
                name="residual_block_0"
            )

            inputs = ops.downsampling2d(
                inputs=inputs,
                factors=[2, 2],
                data_format=self.data_format
            )

            return inputs

    def color_block(self, inputs, index, training, name="color_block", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            inputs = ops.conv2d(
                inputs=inputs,
                filters=self.max_filters >> (index - 1),
                kernel_size=[3, 3],
                strides=[1, 1],
                data_format=self.data_format,
                apply_spectral_normalization=True,
                name="conv2d_0"
            )

            inputs = tf.nn.leaky_relu(inputs)

            return inputs

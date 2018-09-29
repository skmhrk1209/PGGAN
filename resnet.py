from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import collections
import ops


class Generator(object):

    ResidualParam = collections.namedtuple("ResidualParam", ("filters", "blocks"))

    def __init__(self, image_size, filters, residual_params, data_format):

        self.image_size = image_size
        self.filters = filters
        self.residual_params = residual_params
        self.data_format = data_format

    def __call__(self, inputs, training, name="generator", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            seed_size = (np.array(self.image_size) >> len(self.residual_params)).tolist()

            inputs = ops.dense(
                inputs=inputs,
                units=np.prod(seed_size) * self.filters,
                name="dense_0"
            )

            inputs = tf.reshape(
                tensor=inputs,
                shape=[-1] + seed_size + [self.filters]
            )

            if self.data_format == "channels_first":

                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            for i, residual_param in enumerate(self.residual_params, 1):

                inputs = ops.unpooling2d(
                    inputs=inputs,
                    pool_size=[2, 2],
                    data_format=self.data_format
                )

                for j in range(residual_param.blocks):

                    inputs = ops.residual_block(
                        inputs=inputs,
                        filters=residual_param.filters,
                        strides=[1, 1],
                        normalization=ops.batch_normalization,
                        activation=tf.nn.relu,
                        data_format=self.data_format,
                        training=training,
                        name="residual_block_{}_{}".format(i, j)
                    )

            inputs = ops.batch_normalization(
                inputs=inputs,
                data_format=self.data_format,
                training=training,
                name="batch_normalization_{}".format(len(self.residual_params) + 1)
            )

            inputs = tf.nn.relu(inputs)

            inputs = ops.conv2d(
                inputs=inputs,
                filters=3,
                kernel_size=[3, 3],
                strides=[1, 1],
                data_format=self.data_format,
                name="conv2d_{}".format(len(self.residual_params) + 1)
            )

            inputs = tf.nn.sigmoid(inputs)

            return inputs


class Discriminator(object):

    ResidualParam = collections.namedtuple("ResidualParam", ("filters", "blocks"))

    def __init__(self, filters, residual_params, data_format):

        self.filters = filters
        self.residual_params = residual_params
        self.data_format = data_format

    def __call__(self, inputs, training, name="discriminator", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            inputs = ops.conv2d(
                inputs=inputs,
                filters=self.filters,
                kernel_size=[3, 3],
                strides=[1, 1],
                data_format=self.data_format,
                name="conv2d_{}".format(len(self.residual_params) + 1),
                apply_spectral_normalization=True
            )

            for i, residual_param in enumerate(self.residual_params, 1):

                for j, _ in enumerate(range(residual_param.blocks), 1):

                    inputs = ops.residual_block(
                        inputs=inputs,
                        filters=residual_param.filters,
                        strides=[1, 1],
                        normalization=None,
                        activation=tf.nn.relu,
                        data_format=self.data_format,
                        training=training,
                        name="residual_block_{}_{}".format(
                            len(self.residual_params) + 1 - i,
                            len(residual_param.blocks) - j
                        ),
                        apply_spectral_normalization=True
                    )

                inputs = tf.layers.average_pooling2d(
                    inputs=inputs,
                    pool_size=[2, 2],
                    strides=[2, 1],
                    padding="same",
                    data_format=self.data_format
                )

            inputs = tf.nn.relu(inputs)

            inputs = ops.global_average_pooling2d(
                inputs=inputs,
                data_format=self.data_format
            )

            inputs = ops.dense(
                inputs=inputs,
                units=1,
                name="dense_0",
                apply_spectral_normalization=True
            )

            return inputs

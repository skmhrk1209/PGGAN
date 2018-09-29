from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import collections
import ops


class Generator(object):

    DeconvParam = collections.namedtuple("DeconvParam", ("filters"))

    def __init__(self, image_size, filters, deconv_params, data_format):

        self.image_size = image_size
        self.filters = filters
        self.deconv_params = deconv_params
        self.data_format = data_format

    def __call__(self, inputs, training, name="generator", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            seed_size = (np.array(self.image_size) >> len(self.deconv_params)).tolist()

            inputs = ops.dense(
                inputs=inputs,
                units=np.prod(seed_size) * self.filters,
                name="dense_0"
            )

            inputs = ops.batch_normalization(
                inputs=inputs,
                data_format=self.data_format,
                training=training,
                name="batch_normalization_0"
            )

            inputs = tf.nn.relu(inputs)

            inputs = tf.reshape(
                tensor=inputs,
                shape=[-1] + seed_size + [self.filters]
            )

            if self.data_format == "channels_first":

                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            for i, deconv_param in enumerate(self.deconv_params):

                inputs = ops.deconv2d(
                    inputs=inputs,
                    filters=deconv_param.filters,
                    kernel_size=[4, 4],
                    strides=[2, 2],
                    data_format=self.data_format,
                    name="deconv2d_{}".format(i)
                )

                inputs = ops.batch_normalization(
                    inputs=inputs,
                    data_format=self.data_format,
                    training=training,
                    name="batch_normalization_{}".format(i)
                )

                inputs = tf.nn.relu(inputs)

            inputs = ops.deconv2d(
                inputs=inputs,
                filters=3,
                kernel_size=[3, 3],
                strides=[1, 1],
                data_format=self.data_format,
                name="last_deconv2d_{}".format(len(self.deconv_params))
            )

            inputs = tf.nn.sigmoid(inputs)

            return inputs


class Discriminator(object):

    ConvParam = collections.namedtuple("ConvParam", ("filters"))

    def __init__(self, filters, conv_params, data_format):

        self.filters = filters
        self.conv_params = conv_params
        self.data_format = data_format

    def __call__(self, inputs, training, name="discriminator", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            inputs = ops.conv2d(
                inputs=inputs,
                filters=self.filters,
                kernel_size=[3, 3],
                strides=[1, 1],
                data_format=self.data_format,
                name="first_conv2d_{}".format(len(self.conv_params)),
                apply_spectral_normalization=True
            )

            inputs = tf.nn.leaky_relu(inputs)

            for i, conv_param in enumerate(self.conv_params):

                inputs = ops.conv2d(
                    inputs=inputs,
                    filters=conv_param.filters,
                    kernel_size=[4, 4],
                    strides=[2, 2],
                    data_format=self.data_format,
                    name="conv2d_{}_1".format(len(self.conv_params) - 1 - i),
                    apply_spectral_normalization=True
                )

                inputs = tf.nn.leaky_relu(inputs)

                inputs = ops.conv2d(
                    inputs=inputs,
                    filters=conv_param.filters,
                    kernel_size=[3, 3],
                    strides=[1, 1],
                    data_format=self.data_format,
                    name="conv2d_{}_0".format(len(self.conv_params) - 1 - i),
                    apply_spectral_normalization=True
                )

                inputs = tf.nn.leaky_relu(inputs)

            inputs = tf.layers.flatten(inputs)

            inputs = ops.dense(
                inputs=inputs,
                units=1,
                name="dense_0",
                apply_spectral_normalization=True
            )

            return inputs

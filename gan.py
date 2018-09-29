from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import collections
import os
import itertools
import time
import cv2
import utils


class Model(object):

    HyperParam = collections.namedtuple(
        "HyperParam", (
            "latent_size",
            "gradient_coefficient",
            "learning_rate",
            "beta1",
            "beta2"
        )
    )

    def __init__(self, model_dir, dataset, generator, discriminator, hyper_param, name="gan", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            self.model_dir = model_dir
            self.name = name

            self.dataset = dataset
            self.generator = generator
            self.discriminator = discriminator
            self.hyper_param = hyper_param

            self.batch_size = tf.placeholder(
                dtype=tf.int32,
                shape=[],
                name="batch_size"
            )
            self.training = tf.placeholder(
                dtype=tf.bool,
                shape=[],
                name="training"
            )

            self.next_reals = self.dataset.get_next()
            self.next_latents = tf.random_normal(
                shape=[self.batch_size, self.hyper_param.latent_size],
                dtype=tf.float32
            )

            self.reals = tf.placeholder(
                dtype=tf.float32,
                shape=self.next_reals.shape,
                name="reals"
            )
            self.latents = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.hyper_param.latent_size],
                name="latents"
            )

            self.fakes = self.generator(
                inputs=self.latents,
                training=self.training,
                name="generator"
            )

            self.real_logits = self.discriminator(
                inputs=self.reals,
                training=self.training,
                name="discriminator"
            )
            self.fake_logits = self.discriminator(
                inputs=self.fakes,
                training=self.training,
                name="discriminator",
                reuse=True
            )

            self.generator_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.fake_logits,
                    labels=tf.ones_like(self.fake_logits)
                )
            )

            self.discriminator_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.real_logits,
                    labels=tf.ones_like(self.real_logits)
                )
            )
            self.discriminator_loss += tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.fake_logits,
                    labels=tf.zeros_like(self.fake_logits)
                )
            )

            self.interpolate_coefficients = tf.random_uniform(
                shape=[self.batch_size, 1, 1, 1],
                dtype=tf.float32
            )
            self.interpolates = self.reals + (self.fakes - self.reals) * self.interpolate_coefficients
            self.interpolate_logits = self.discriminator(
                inputs=interpolates,
                training=self.training,
                name="discriminator",
                reuse=True
            )

            self.gradients = tf.gradients(ys=self.interpolate_logits, xs=self.interpolates)[0]
            self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), axis=[1, 2, 3]) + 0.0001)
            self.gradient_penalty = tf.reduce_mean(tf.square(self.slopes - 1.0))
            self.discriminator_loss += self.gradient_penalty * self.hyper_param.gradient_coefficient

            self.generator_variables = tf.get_collection(
                key=tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="{}/generator".format(self.name)
            )
            self.discriminator_variables = tf.get_collection(
                key=tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="{}/discriminator".format(self.name)
            )

            self.generator_global_step = tf.Variable(initial_value=0, trainable=False)
            self.discriminator_global_step = tf.Variable(initial_value=0, trainable=False)

            self.generator_optimizer = tf.train.AdamOptimizer(
                learning_rate=self.hyper_param.learning_rate,
                beta1=self.hyper_param.beta1,
                beta2=self.hyper_param.beta2
            )
            self.discriminator_optimizer = tf.train.AdamOptimizer(
                learning_rate=self.hyper_param.learning_rate,
                beta1=self.hyper_param.beta1,
                beta2=self.hyper_param.beta2
            )

            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(self.update_ops):

                self.generator_train_op = self.generator_optimizer.minimize(
                    loss=self.generator_loss,
                    var_list=self.generator_variables,
                    global_step=self.generator_global_step
                )

                self.discriminator_train_op = self.discriminator_optimizer.minimize(
                    loss=self.discriminator_loss,
                    var_list=self.discriminator_variables,
                    global_step=self.discriminator_global_step
                )

            self.saver = tf.train.Saver()

    def initialize(self):

        session = tf.get_default_session()

        checkpoint = tf.train.latest_checkpoint(self.model_dir)

        if checkpoint:
            self.saver.restore(session, checkpoint)
            print(checkpoint, "loaded")

        else:
            global_variables = tf.global_variables(scope=self.name)
            session.run(tf.variables_initializer(global_variables))
            print("global variables in {} initialized".format(self.name))

    def reinitialize(self):

        session = tf.get_default_session()

        uninitialized_variables = [
            variable
            for variable in tf.global_variables(self.name)
            if not session.run(tf.is_variable_initialized(variable))
        ]

        session.run(tf.variables_initializer(uninitialized_variables))
        print("uninitialized variables in {} initialized".format(self.name))

    def train(self, filenames, batch_size, num_epochs, buffer_size):

        session = tf.get_default_session()

        try:

            print("training started")

            start = time.time()

            self.dataset.initialize(
                filenames=filenames,
                batch_size=batch_size,
                num_epochs=num_epochs,
                buffer_size=buffer_size
            )

            for i in itertools.count():

                feed_dict = {self.batch_size: batch_size}

                reals, latents = session.run(
                    [self.next_reals, self.next_latents],
                    feed_dict=feed_dict
                )

                feed_dict.update({
                    self.latents: latents,
                    self.reals: reals
                })

                training_placeholder_names = [
                    "{}:0".format(operation.name)
                    for operation in tf.get_default_graph().get_operations()
                    if "training" in operation.name
                ]

                training_placeholders = [
                    tf.get_default_graph().get_tensor_by_name(training_placeholder_name)
                    for training_placeholder_name in training_placeholder_names
                ]

                feed_dict.update({
                    training_placeholder: True
                    for training_placeholder in training_placeholders
                })

                session.run(self.generator_train_op, feed_dict=feed_dict)
                session.run(self.discriminator_train_op, feed_dict=feed_dict)

                if i % 100 == 0:

                    generator_global_step, generator_loss = session.run(
                        [self.generator_global_step, self.generator_loss],
                        feed_dict=feed_dict
                    )

                    print("global_step: {}, generator_loss: {:.2f}".format(
                        generator_global_step,
                        generator_loss
                    ))

                    discriminator_global_step, discriminator_loss = session.run(
                        [self.discriminator_global_step, self.discriminator_loss],
                        feed_dict=feed_dict
                    )

                    print("global_step: {}, discriminator_loss: {:.2f}".format(
                        discriminator_global_step,
                        discriminator_loss
                    ))

                    checkpoint = self.saver.save(
                        sess=session,
                        save_path=os.path.join(self.model_dir, "model.ckpt"),
                        global_step=generator_global_step
                    )

                    stop = time.time()

                    print("{} saved ({:.2f} sec)".format(checkpoint, stop - start))

                    start = time.time()

                    if i % 1000 == 0:

                        fakes = session.run(self.fakes, feed_dict=feed_dict)

                        images = np.concatenate([reals, fakes], axis=2)

                        images = utils.scale(images, 0, 1, 0, 255)

                        for j, image in enumerate(images):

                            cv2.imwrite(
                                "generated/image_{}_{}.png".format(i, j),
                                cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            )

        except tf.errors.OutOfRangeError:

            print("training ended")

    def predict(self, filenames, batch_size, num_epochs, buffer_size):

        session = tf.get_default_session()

        try:

            print("prediction started")

            self.dataset.initialize(
                filenames=filenames,
                batch_size=batch_size,
                num_epochs=num_epochs,
                buffer_size=buffer_size
            )

            for i in itertools.count():

                feed_dict = {self.batch_size: batch_size}

                reals, latents = session.run(
                    [self.next_reals, self.next_latents],
                    feed_dict=feed_dict
                )

                feed_dict.update({
                    self.latents: latents,
                    self.training: False
                })

                fakes = session.run(self.fakes, feed_dict=feed_dict)

                images = np.concatenate([reals, fakes], axis=2)

                for image in images:

                    cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                    cv2.waitKey(100)

        except tf.errors.OutOfRangeError:

            print("prediction ended")

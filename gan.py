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

    def __init__(self, dataset, generator, discriminator, hyper_param, name="gan", reuse=None):

        # if train this model in PGGAN style
        # set reuse=tf.AUTO_REUSE
        with tf.variable_scope(name, reuse=reuse):

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

            ### [CAUTION] ###
            # if assign get_next() to input data tensor,
            # running any operation that depends on input data tensor
            # advances input data iterator!
            # so, use placeholder for reals
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

            self.fakes = generator(
                inputs=self.latents,
                training=self.training,
                name="generator"
            )

            self.real_logits = discriminator(
                inputs=self.reals,
                training=self.training,
                name="discriminator"
            )
            self.fake_logits = discriminator(
                inputs=self.fakes,
                training=self.training,
                name="discriminator",
                reuse=True
            )

            # minimize JS Divergence(GAN)
            # instead of Wasserstein Distance(WGAN)
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

            # add WGAN-GP gradient penalty to discriminator loss
            # slopes throws NaN (https://github.com/tdeboissiere/DeepLearningImplementations/issues/68)
            # so add epsilon inside sqrt()
            self.interpolate_coefficients = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], dtype=tf.float32)
            self.interpolates = self.reals + (self.fakes - self.reals) * self.interpolate_coefficients
            self.interpolate_logits = discriminator(inputs=self.interpolates, training=self.training, name="discriminator", reuse=True)

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

            # it's ok generator global step and discriminator global step isn't same
            self.generator_global_step = tf.get_variable(
                name="generator_global_step",
                shape=[],
                dtype=tf.int32,
                initializer=tf.zeros_initializer(),
                trainable=False
            )
            self.discriminator_global_step = tf.get_variable(
                name="discriminator_global_step",
                shape=[],
                dtype=tf.int32,
                initializer=tf.zeros_initializer(),
                trainable=False
            )

            # tune hyper parameter learning rate, beta1, beta2
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

            # to update moving_mean and moving_variance for batch normalization when trainig,
            # run update operation before run train operation
            # this update operation is placed in tf.GraphKeys.UPDATE_OPS
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

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

            # save variables already defined
            self.saver = tf.train.Saver()

            # setup summaries of important tensor
            self.summary = tf.summary.merge([
                tf.summary.image("reals", self.reals),
                tf.summary.image("fakes", self.fakes),
                tf.summary.scalar("generator_loss", self.generator_loss),
                tf.summary.scalar("discriminator_loss", self.discriminator_loss),
                tf.summary.scalar("gradient_penalty", self.gradient_penalty),
            ])

    # call this when train model untrained or still training
    # in this case, model can restore variables from checkpoint.
    def initialize(self):

        session = tf.get_default_session()

        checkpoint = tf.train.latest_checkpoint(self.name)

        if checkpoint:
            self.saver.restore(session, checkpoint)
            print(checkpoint, "loaded")

        else:
            global_variables = tf.global_variables(scope=self.name)
            session.run(tf.variables_initializer(global_variables))
            print("global variables in {} initialized".format(self.name))

    # call this when train model using pre-trained model
    # in this case, initialize only uninitialized variables
    def reinitialize(self):

        session = tf.get_default_session()

        uninitialized_variables = [
            variable for variable in tf.global_variables(self.name)
            if not session.run(tf.is_variable_initialized(variable))
        ]

        session.run(tf.variables_initializer(uninitialized_variables))
        print("uninitialized variables in {} initialized".format(self.name))

    def train(self, filenames, batch_size, num_epochs, buffer_size):

        session = tf.get_default_session()
        writer = tf.summary.FileWriter(self.name, session.graph)

        try:

            print("training started")

            start = time.time()

            # initialize dataset iterator
            self.dataset.initialize(
                filenames=filenames,
                batch_size=batch_size,
                num_epochs=num_epochs,
                buffer_size=buffer_size
            )

            ### [CAUTION] ###
            # variables in pre-trained model depends placeholders that doesn't exist in this instance.
            # so, search those placeholders in graph, and feed values to them.
            latents_placeholder_names = [
                "{}:0".format(operation.name)
                for operation in tf.get_default_graph().get_operations()
                if "latents" in operation.name
            ]

            training_placeholder_names = [
                "{}:0".format(operation.name)
                for operation in tf.get_default_graph().get_operations()
                if "training" in operation.name
            ]

            latents_placeholders = [
                tf.get_default_graph().get_tensor_by_name(latents_placeholder_name)
                for latents_placeholder_name in latents_placeholder_names
            ]

            training_placeholders = [
                tf.get_default_graph().get_tensor_by_name(training_placeholder_name)
                for training_placeholder_name in training_placeholder_names
            ]

            for i in itertools.count():

                feed_dict = {self.batch_size: batch_size}

                reals, latents = session.run(
                    fetches=[self.next_reals, self.next_latents],
                    feed_dict=feed_dict
                )

                feed_dict.update({self.reals: reals})

                feed_dict.update({
                    latents_placeholder: latents
                    for latents_placeholder in latents_placeholders
                })

                feed_dict.update({
                    training_placeholder: True
                    for training_placeholder in training_placeholders
                })

                session.run(
                    fetches=[self.generator_train_op, self.discriminator_train_op],
                    feed_dict=feed_dict
                )

                if i % 100 == 0:

                    generator_global_step, generator_loss = session.run(
                        fetches=[self.generator_global_step, self.generator_loss],
                        feed_dict=feed_dict
                    )
                    print("global_step: {}, generator_loss: {:.2f}".format(
                        generator_global_step,
                        generator_loss
                    ))

                    discriminator_global_step, discriminator_loss = session.run(
                        fetches=[self.discriminator_global_step, self.discriminator_loss],
                        feed_dict=feed_dict
                    )
                    print("global_step: {}, discriminator_loss: {:.2f}".format(
                        discriminator_global_step,
                        discriminator_loss
                    ))

                    summary = session.run(self.summary, feed_dict=feed_dict)
                    writer.add_summary(summary, global_step=generator_global_step)

                    fakes = session.run(self.fakes, feed_dict=feed_dict)
                    images = np.concatenate([reals, fakes], axis=2)
                    # images = [utils.scale(image, 0, 1, 0, 255) for image in images]
                    images = [cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in images]

                    for j, image in enumerate(images):

                        cv2.imshow("image", image)
                        cv2.waitKey(100)
                        # cv2.imwrite("generated/image_{}_{}.png".format(i, j), image)

                    if i % 1000 == 0:

                        checkpoint = self.saver.save(
                            sess=session,
                            save_path=os.path.join(self.name, "model.ckpt"),
                            global_step=generator_global_step
                        )

                        stop = time.time()
                        print("{} saved ({:.2f} sec)".format(checkpoint, stop - start))
                        start = time.time()

        except tf.errors.OutOfRangeError:

            print("training ended")

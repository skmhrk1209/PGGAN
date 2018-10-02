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


class Model(object):

    #=============================================================#
    # enum for loss function type
    #=============================================================#
    class LossFunction:
        NS_GAN, WGAN = range(2)

    #=============================================================#
    # enum for gradient penalty type
    #=============================================================#
    class GradientPenalty:
        ZERO_CENTERED, ONE_CENTERED = range(2)

    #=============================================================#
    # model difinition
    #=============================================================#
    def __init__(self, dataset, generator, discriminator, loss_function,
                 gradient_penalty, hyper_parameters, name="gan", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            self.name = name
            self.dataset = dataset
            self.generator = generator
            self.discriminator = discriminator
            self.hyper_parameters = hyper_parameters

            self.batch_size = tf.placeholder(
                dtype=tf.int32,
                shape=[]
            )
            self.training = tf.placeholder(
                dtype=tf.bool,
                shape=[]
            )

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

            #=============================================================#
            # "coloring_index" for progressive growing architecture
            #=============================================================#
            self.coloring_index = self.hyper_parameters.coloring_index_fn(
                global_step=tf.cast(self.discriminator_global_step, tf.float32)
            )

            self.next_reals = self.dataset.get_next()

            self.next_latents = tf.random_normal(
                shape=[self.batch_size, self.hyper_parameters.latent_size],
                dtype=tf.float32
            )

            self.reals = tf.placeholder(
                dtype=tf.float32,
                shape=self.next_reals.shape
            )

            self.latents = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.hyper_parameters.latent_size]
            )

            self.fakes = generator(
                inputs=self.latents,
                coloring_index=self.coloring_index,
                training=self.training,
                name="generator"
            )

            self.real_logits = discriminator(
                inputs=self.reals,
                coloring_index=self.coloring_index,
                training=self.training,
                name="discriminator"
            )
            self.fake_logits = discriminator(
                inputs=self.fakes,
                coloring_index=self.coloring_index,
                training=self.training,
                name="discriminator",
                reuse=True
            )

            #=============================================================#
            # NS-GAN loss function
            #=============================================================#
            if loss_function == Model.LossFunction.NS_GAN:

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

            #=============================================================#
            # WGAN loss function
            #=============================================================#
            elif loss_function == Model.LossFunction.WGAN:

                self.generator_loss = -tf.reduce_mean(self.fake_logits)

                self.discriminator_loss = -tf.reduce_mean(self.real_logits)
                self.discriminator_loss += tf.reduce_mean(self.fake_logits)

            else:
                raise ValueError("Invalid loss function")

            #=============================================================#
            # interpolation for gradient penalty
            # to avoid NaN exception, add epsilon inside sqrt()
            # (https://github.com/tdeboissiere/DeepLearningImplementations/issues/68)
            #=============================================================#
            self.interpolate_coefficients = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], dtype=tf.float32)
            self.interpolates = self.reals + (self.fakes - self.reals) * self.interpolate_coefficients
            self.interpolate_logits = discriminator(
                inputs=self.interpolates,
                training=self.training,
                coloring_index=self.coloring_index,
                name="discriminator",
                reuse=True
            )

            self.gradients = tf.gradients(ys=self.interpolate_logits, xs=self.interpolates)[0]
            self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), axis=[1, 2, 3]) + 0.0001)

            #=============================================================#
            # zero-centered gradient penalty
            # (https://openreview.net/pdf?id=ByxPYjC5KQ)
            #=============================================================#
            if gradient_penalty == Model.GradientPenalty.ZERO_CENTERED:

                self.gradient_penalty = tf.reduce_mean(tf.square(self.slopes - 0.0))

            #=============================================================#
            # one-centered gradient penalty (WGAN-GP)
            #=============================================================#
            elif gradient_penalty == Model.GradientPenalty.ONE_CENTERED:

                self.gradient_penalty = tf.reduce_mean(tf.square(self.slopes - 1.0))

            else:
                raise ValueError("Invalid gradient penalty")

            self.discriminator_loss += self.gradient_penalty * self.hyper_parameters.gradient_coefficient

            self.generator_variables = tf.get_collection(
                key=tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="{}/generator".format(self.name)
            )
            self.discriminator_variables = tf.get_collection(
                key=tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="{}/discriminator".format(self.name)
            )

            #=============================================================#
            # tune hyper parameter learning rate, beta1, beta2
            #=============================================================#
            self.generator_optimizer = tf.train.AdamOptimizer(
                learning_rate=self.hyper_parameters.learning_rate,
                beta1=self.hyper_parameters.beta1,
                beta2=self.hyper_parameters.beta2
            )
            self.discriminator_optimizer = tf.train.AdamOptimizer(
                learning_rate=self.hyper_parameters.learning_rate,
                beta1=self.hyper_parameters.beta1,
                beta2=self.hyper_parameters.beta2
            )

            #=============================================================#
            # to update moving_mean and moving_variance
            # for batch normalization when trainig,
            # run update operation before run train operation
            # this update operation is placed in tf.GraphKeys.UPDATE_OPS
            #=============================================================#
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

            #=============================================================#
            # save variables already defined to checkpoint
            #=============================================================#
            self.saver = tf.train.Saver()

            #=============================================================#
            # setup summaries of important tensor
            #=============================================================#
            self.summary = tf.summary.merge([
                tf.summary.image("reals", self.reals, max_outputs=10),
                tf.summary.image("fakes", self.fakes, max_outputs=10),
                tf.summary.scalar("generator_loss", self.generator_loss),
                tf.summary.scalar("discriminator_loss", self.discriminator_loss),
                tf.summary.scalar("gradient_penalty", self.gradient_penalty),
            ])

    #=============================================================#
    # restore variables from checkpoint
    # or initialize global variables
    #=============================================================#
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

    #=============================================================#
    # training step
    #=============================================================#
    def train(self, filenames, num_epochs, batch_size, buffer_size):

        session = tf.get_default_session()
        writer = tf.summary.FileWriter(self.name, session.graph)

        print("training started")

        start = time.time()

        #=============================================================#
        # initialize dataset iterator
        #=============================================================#
        self.dataset.initialize(
            filenames=filenames,
            num_epochs=num_epochs,
            batch_size=batch_size,
            buffer_size=buffer_size
        )

        for i in itertools.count():

            feed_dict = {self.batch_size: batch_size}

            try:
                reals, latents = session.run(
                    [self.next_reals, self.next_latents],
                    feed_dict=feed_dict
                )

            except tf.errors.OutOfRangeError:
                print("training ended")
                break

            else:
                if reals.shape[0] != batch_size:
                    break

            feed_dict.update({
                self.reals: reals,
                self.latents: latents
            })

            session.run(
                [self.generator_train_op, self.discriminator_train_op],
                feed_dict=feed_dict
            )

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

                coloring_index = session.run(self.coloring_index)
                print("coloring_index: {:2f}".format(coloring_index))

                summary = session.run(self.summary, feed_dict=feed_dict)
                writer.add_summary(summary, global_step=generator_global_step)

                if i % 100000 == 0:

                    checkpoint = self.saver.save(
                        sess=session,
                        save_path=os.path.join(self.name, "model.ckpt"),
                        global_step=generator_global_step
                    )

                    stop = time.time()
                    print("{} saved ({:.2f} sec)".format(checkpoint, stop - start))
                    start = time.time()

                    fakes = session.run(self.fakes, feed_dict=feed_dict)
                    images = np.concatenate([reals, fakes], axis=2)
                    images *= 255.0

                    for j, image in enumerate(images):

                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite("generated/image_{}_{}.png".format(i, j), image)

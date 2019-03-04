#=================================================================================================#
# TensorFlow implementation of PGGAN
# [Progressive Growing of GANs for Improved Quality, Stability, and Variation]
# (https://arxiv.org/pdf/1710.10196.pdf)
#=================================================================================================#

import tensorflow as tf
import numpy as np
import argparse
import functools
import pickle
from dataset import celeba_input_fn
from model import GAN
from network import PGGAN
from param import Param

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="celeba_style_gan_model")
parser.add_argument('--filenames', type=str, nargs="+", default=["celeba_train.tfrecord"])
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--total_steps", type=int, default=1000000)
parser.add_argument("--train", action="store_true")
parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)

with open("attr_counts.pickle", "rb") as file:
    attr_counts = pickle.load(file)

with tf.Graph().as_default():

    tf.set_random_seed(0)

    pggan = PGGAN(
        min_resolution=[4, 4],
        max_resolution=[256, 256],
        min_channels=16,
        max_channels=512
    )

    gan = GAN(
        discriminator=pggan.discriminator,
        generator=pggan.generator,
        real_input_fn=functools.partial(
            celeba_input_fn,
            filenames=args.filenames,
            batch_size=args.batch_size,
            num_epochs=None,
            shuffle=True,
            image_size=[256, 256]
        ),
        fake_input_fn=lambda: (
            tf.random_normal([args.batch_size, 512]),
            tf.one_hot(tf.reshape(tf.random.multinomial(
                logits=tf.log([tf.cast(attr_counts, tf.float32)]),
                num_samples=args.batch_size
            ), [args.batch_size]), len(attr_counts))
        ),
        hyper_params=Param(
            discriminator_learning_rate=1e-3,
            discriminator_beta1=0.0,
            discriminator_beta2=0.99,
            generator_learning_rate=1e-3,
            generator_beta1=0.0,
            generator_beta2=0.99,
            r1_gamma=10.0,
            r2_gamma=0.0
        ),
        name=args.model_dir
    )

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list=args.gpu,
            allow_growth=True
        )
    )

    with tf.Session(config=config) as session:

        gan.initialize()
        gan.train(args.total_steps)

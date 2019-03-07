#=================================================================================================#
# TensorFlow implementation of PGGAN
# [Progressive Growing of GANs for Improved Quality, Stability, and Variation]
# (https://arxiv.org/pdf/1710.10196.pdf)
#=================================================================================================#

import tensorflow as tf
import argparse
import functools
from dataset import celeba_input_fn
from model import GAN
from network import PGGAN
from param import Param

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="celeba_pggan_model")
parser.add_argument('--filenames', type=str, nargs="+", default=["celeba_train.tfrecord"])
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--total_steps", type=int, default=1000000)
parser.add_argument("--train", action="store_true")
parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)

with tf.Graph().as_default():

    tf.set_random_seed(0)

    pggan = PGGAN(
        min_resolution=[4, 4],
        max_resolution=[256, 256],
        min_channels=16,
        max_channels=512,
        growing_level=tf.cast(tf.get_variable(
            name="global_step",
            initializer=0,
            trainable=False
        ) / args.total_steps, tf.float32)
    )

    gan = GAN(
        generator=pggan.generator,
        discriminator=pggan.discriminator,
        real_input_fn=functools.partial(
            celeba_input_fn,
            filenames=args.filenames,
            batch_size=args.batch_size,
            num_epochs=None,
            shuffle=True,
            image_size=[256, 256]
        ),
        fake_input_fn=lambda: (
            tf.random_normal([args.batch_size, 512])
        ),
        hyper_params=Param(
            generator_learning_rate=2e-3,
            generator_beta1=0.0,
            generator_beta2=0.99,
            discriminator_learning_rate=2e-3,
            discriminator_beta1=0.0,
            discriminator_beta2=0.99,
            one_centered_gradient_penalty_weight=10.0,
            generator_auxiliary_classification_weight=0.0,
            discriminator_auxiliary_classification_weight=0.0,
        )
    )

    if args.train:

        gan.train(
            total_steps=args.total_steps,
            model_dir=args.model_dir,
            save_checkpoint_steps=1000,
            save_summary_steps=100,
            log_step_count_steps=100,
            config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    visible_device_list=args.gpu,
                    allow_growth=True
                )
            )
        )

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
from utils import Struct

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="celeba_pggan_model")
parser.add_argument("--sample_dir", type=str, default="celeba_pggan_samples")
parser.add_argument('--filenames', type=str, nargs="+", default=["celeba_train.tfrecord"])
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_epochs", type=int, default=None)
parser.add_argument("--total_steps", type=int, default=1000000)
parser.add_argument("--train", action="store_true")
parser.add_argument('--evaluate', action="store_true")
parser.add_argument('--generate', action="store_true")
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
        growing_level=tf.cast(tf.divide(
            x=tf.train.create_global_step(),
            y=args.total_steps
        ), tf.float32)
    )

    gan = GAN(
        generator=pggan.generator,
        discriminator=pggan.discriminator,
        real_input_fn=functools.partial(
            celeba_input_fn,
            filenames=args.filenames,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs if args.train else 1,
            shuffle=True if args.train else False,
            image_size=[256, 256]
        ),
        fake_input_fn=lambda: (
            tf.random_normal([args.batch_size, 512])
        ),
        hyper_params=Struct(
            generator_learning_rate=2e-3,
            generator_beta1=0.0,
            generator_beta2=0.99,
            discriminator_learning_rate=2e-3,
            discriminator_beta1=0.0,
            discriminator_beta2=0.99,
            gradient_penalty_weight=10.0,
            generator_classification_loss_weight=10.0,
            discriminator_classification_loss_weight=10.0,
        )
    )

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list=args.gpu,
            allow_growth=True
        )
    )

    if args.train:
        gan.train(
            model_dir=args.model_dir,
            total_steps=args.total_steps,
            save_checkpoint_steps=10000,
            save_summary_steps=1000,
            log_tensor_steps=1000,
            config=config
        )

    if args.evaluate:
        gan.evaluate(
            model_dir=args.model_dir,
            config=config
        )

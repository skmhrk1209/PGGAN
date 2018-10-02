from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
from models import gan
from archs import dcgan, resnet
from data import celeba

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="celeba_dcgan_model", help="model directory")
parser.add_argument('--filenames', type=str, nargs="+", default=["train.tfrecord"], help="tfrecord filenames")
parser.add_argument("--num_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=10, help="batch size")
parser.add_argument("--buffer_size", type=int, default=100000, help="buffer size to shuffle dataset")
parser.add_argument('--data_format', type=str, choices=["channels_first", "channels_last"],
                    default="channels_last", help="data_format")
parser.add_argument('--architecture', type=str, choices=["dcgan", "resnet"],
                    default="dcgan", help="architecture of generator and discriminator")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument('--predict', action="store_true", help="with prediction")
parser.add_argument('--gpu', type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)

architecture = dcgan if args.architecture == "dcgan" else resnet

gan_model = gan.Model(
    dataset=celeba.Dataset(
        image_size=[128, 128],
        data_format=args.data_format
    ),
    generator=architecture.Generator(
        min_resolution=4,
        max_resolution=128,
        max_filters=512,
        data_format=args.data_format,
    ),
    discriminator=architecture.Discriminator(
        min_resolution=4,
        max_resolution=128,
        max_filters=512,
        data_format=args.data_format
    ),
    hyper_param=gan.Model.HyperParam(
        latent_size=128,
        gradient_coefficient=1.0,
        learning_rate=0.0002,
        beta1=0.5,
        beta2=0.999,
        coloring_index_fn=(
            lambda global_step:
            global_step / 100000.0 + 1.0
        )
    ),
    name=args.model_dir
)

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list=args.gpu,
        allow_growth=True
    ),
    log_device_placement=False,
    allow_soft_placement=True
)

with tf.Session(config=config) as session:

    if args.train:

        gan_model.initialize()

        gan_model.train(
            filenames=args.filenames,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size
        )

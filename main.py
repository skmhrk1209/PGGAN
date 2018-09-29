from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import gan
import dcgan
import resnet
import dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="celeba_dcgan_model", help="model directory")
parser.add_argument("--num_epochs", type=int, nargs="+", default=[2, 4, 8], help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=10, help="batch size")
parser.add_argument("--buffer_size", type=int, default=100000, help="buffer size to shuffle dataset")
parser.add_argument('--data_format', type=str, choices=["channels_first", "channels_last"], default="channels_last", help="data_format")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument('--predict', action="store_true", help="with prediction")
parser.add_argument('--gpu', type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


class Dataset(dataset.Dataset):

    def __init__(self, image_size, data_format):

        self.image_size = image_size
        self.data_format = data_format

        super(Dataset, self).__init__()

    def parse(self, example):

        features = tf.parse_single_example(
            serialized=example,
            features={
                "path": tf.FixedLenFeature(
                    shape=[],
                    dtype=tf.string,
                    default_value=""
                ),
                "label": tf.FixedLenFeature(
                    shape=[],
                    dtype=tf.int64,
                    default_value=0
                )
            }
        )

        image = tf.read_file(features["path"])
        image = tf.image.decode_jpeg(image, 3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_image_with_crop_or_pad(image, 128, 128)
        image = tf.image.resize_images(image, self.image_size)

        if self.data_format == "channels_first":

            image = tf.transpose(image, [2, 0, 1])

        return image


gan_models = [
    gan.Model(
        dataset=Dataset(
            image_size=[32, 32],
            data_format=args.data_format
        ),
        generator=dcgan.Generator(
            image_size=[32, 32],
            filters=512,
            deconv_params=[
                dcgan.Generator.DeconvParam(filters=256),
                # dcgan.Generator.DeconvParam(filters=128),
                # dcgan.Generator.DeconvParam(filters=64)
            ],
            data_format=args.data_format,
        ),
        discriminator=dcgan.Discriminator(
            filters=256,
            conv_params=[
                # dcgan.Discriminator.ConvParam(filters=128),
                # dcgan.Discriminator.ConvParam(filters=256),
                dcgan.Discriminator.ConvParam(filters=512)
            ],
            data_format=args.data_format
        ),
        hyper_param=gan.Model.HyperParam(
            latent_size=128,
            gradient_coefficient=1.0,
            learning_rate=0.0002,
            beta1=0.5,
            beta2=0.999
        ),
        name=args.model_dir
    ),
    gan.Model(
        dataset=Dataset(
            image_size=[64, 64],
            data_format=args.data_format
        ),
        generator=dcgan.Generator(
            image_size=[64, 64],
            filters=512,
            deconv_params=[
                dcgan.Generator.DeconvParam(filters=256),
                dcgan.Generator.DeconvParam(filters=128),
                # dcgan.Generator.DeconvParam(filters=64)
            ],
            data_format=args.data_format,
        ),
        discriminator=dcgan.Discriminator(
            filters=128,
            conv_params=[
                # dcgan.Discriminator.ConvParam(filters=128),
                dcgan.Discriminator.ConvParam(filters=256),
                dcgan.Discriminator.ConvParam(filters=512)
            ],
            data_format=args.data_format
        ),
        hyper_param=gan.Model.HyperParam(
            latent_size=128,
            gradient_coefficient=1.0,
            learning_rate=0.0002,
            beta1=0.5,
            beta2=0.999
        ),
        name=args.model_dir,
        reuse=tf.AUTO_REUSE
    ),
    gan.Model(
        dataset=Dataset(
            image_size=[128, 128],
            data_format=args.data_format
        ),
        generator=dcgan.Generator(
            image_size=[128, 128],
            filters=512,
            deconv_params=[
                dcgan.Generator.DeconvParam(filters=256),
                dcgan.Generator.DeconvParam(filters=128),
                dcgan.Generator.DeconvParam(filters=64)
            ],
            data_format=args.data_format,
        ),
        discriminator=dcgan.Discriminator(
            filters=64,
            conv_params=[
                dcgan.Discriminator.ConvParam(filters=128),
                dcgan.Discriminator.ConvParam(filters=256),
                dcgan.Discriminator.ConvParam(filters=512)
            ],
            data_format=args.data_format
        ),
        hyper_param=gan.Model.HyperParam(
            latent_size=128,
            gradient_coefficient=1.0,
            learning_rate=0.0002,
            beta1=0.5,
            beta2=0.999
        ),
        name=args.model_dir,
        reuse=tf.AUTO_REUSE
    ),
]

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

        for i, (gan_model, num_epochs) in enumerate(zip(gan_models, args.num_epochs)):

            gan_model.reinitialize() if i else gan_model.initialize()

            if i == 0: continue

            gan_model.train(
                filenames=["data/train.tfrecord"],
                num_epochs=num_epochs,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size
            )

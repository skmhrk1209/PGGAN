import tensorflow as tf
import numpy as np
import pickle
import os
from utils import Struct


def cifar10_input_fn(filenames, batch_size, num_epochs, shuffle):

    def unpickle(file):
        with open(file, "rb") as file:
            dict = pickle.load(file, encoding="bytes")
        return dict

    def preprocess(images, labels):

        def normalize(inputs, mean, std):
            return (inputs - mean) / std

        images = tf.reshape(images, [-1, 3, 32, 32])
        images = tf.image.convert_image_dtype(images, tf.float32)
        images = tf.image.random_flip_left_right(images)
        images = normalize(images, 0.5, 0.5)

        labels = tf.cast(labels, tf.int32)
        labels = tf.one_hot(labels, 10)

        return images, labels

    dicts = [unpickle(filename) for filename in filenames]
    images = np.concatenate([dict[b"data"] for dict in dicts])
    labels = np.concatenate([dict[b"labels"] for dict in dicts])

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(images),
            reshuffle_each_iteration=True
        )
    dataset = dataset.repeat(count=num_epochs)
    dataset = dataset.batch(
        batch_size=batch_size,
        drop_remainder=True
    )
    dataset = dataset.map(
        map_func=preprocess,
        num_parallel_calls=os.cpu_count()
    )
    dataset = dataset.prefetch(buffer_size=1)

    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()


def celeba_input_fn(filenames, batch_size, num_epochs, shuffle, image_size):

    def parse_example(example):

        features = Struct(tf.parse_single_example(
            serialized=example,
            features=dict(path=tf.FixedLenFeature([], dtype=tf.string))
        ))

        image = tf.read_file(features.path)
        image = tf.image.decode_jpeg(image, 3)

        return image

    def preprocess(images):

        def normalize(inputs, mean, std):
            return (inputs - mean) / std

        images = tf.image.convert_image_dtype(images, tf.float32)
        images = tf.image.resize_images(images, image_size)
        images = tf.image.random_flip_left_right(images)
        images = tf.transpose(images, [0, 3, 1, 2])
        images = normalize(images, 0.5, 0.5)

        return images

    dataset = tf.data.TFRecordDataset(filenames)
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=sum([
                len(list(tf.io.tf_record_iterator(filename)))
                for filename in filenames
            ]),
            reshuffle_each_iteration=True
        )
    dataset = dataset.repeat(count=num_epochs)
    dataset = dataset.map(
        map_func=parse_example,
        num_parallel_calls=os.cpu_count()
    )
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.map(
        map_func=preprocess,
        num_parallel_calls=os.cpu_count()
    )
    dataset = dataset.prefetch(buffer_size=1)

    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()

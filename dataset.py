import tensorflow as tf
import os


def celeba_input_fn(filenames, batch_size, num_epochs, shuffle, image_size):

    def parse_example(example):
        features = tf.parse_single_example(
            serialized=example,
            features={
                "path": tf.FixedLenFeature([], dtype=tf.string),
                "label": tf.FixedLenFeature([40], dtype=tf.int64),
            }
        )

        image = tf.read_file(features["path"])
        image = tf.image.decode_jpeg(image, 3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, image_size)
        image = tf.transpose(image, [2, 0, 1])
        image = image * 2.0 - 1.0

        return image

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
    dataset = dataset.prefetch(buffer_size=1)

    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()

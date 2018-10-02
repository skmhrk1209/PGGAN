import tensorflow as tf
import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("directory", type=str)
args = parser.parse_args()

with tf.python_io.TFRecordWriter(args.filename) as writer:

    for file in glob.glob(os.path.join(args.directory, "*")):

        writer.write(
            record=tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "path": tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[file.encode("utf-8")]
                            )
                        )
                    }
                )
            ).SerializeToString()
        )

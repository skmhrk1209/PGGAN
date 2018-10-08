import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os

modules = [
    "https://tfhub.dev/google/compare_gan/model_1_celebahq128_resnet19/1",
    "https://tfhub.dev/google/compare_gan/model_2_celebahq128_resnet19/1",
    "https://tfhub.dev/google/compare_gan/model_5_celebahq128_resnet19/1",
    "https://tfhub.dev/google/compare_gan/model_6_celebahq128_resnet19/1",
    "https://tfhub.dev/google/compare_gan/model_9_celebahq128_resnet19/1",
    "https://tfhub.dev/google/progan-128/1"
]

for module in modules[:-1]:

    gan = hub.Module(module)

    latents = tf.placeholder(tf.float32, [None, 128], name="latents")
    fakes = tf.identity(gan(latents, signature="generator"), name="fakes")

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())
        print(session.run(fakes, feed_dict={latents: np.random.uniform(low=-1, high=1, size=[64, 128])}))

        checkpoint = tf.train.Saver().save(
            sess=session,
            save_path=os.path.join(module.split("/")[-2], "model.ckpt")
        )

        tf.train.write_graph(
            graph_or_graph_def=session.graph.as_graph_def(),
            logdir=module.split("/")[-2],
            name="graph.pbtxt",
            as_text=True
        )

for module in modules[-1:]:

    gan = hub.Module(module)

    latents = tf.placeholder(tf.float32, [None, 512], name="latents")
    fakes = tf.identity(gan(latents), name="fakes")

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())
        print(session.run(fakes, feed_dict={latents: np.random.normal(loc=0, scale=1, size=[16, 512])}))

        checkpoint = tf.train.Saver().save(
            sess=session,
            save_path=os.path.join(module.split("/")[-2], "model.ckpt")
        )

        tf.train.write_graph(
            graph_or_graph_def=session.graph.as_graph_def(),
            logdir=module.split("/")[-2],
            name="graph.pbtxt",
            as_text=True
        )

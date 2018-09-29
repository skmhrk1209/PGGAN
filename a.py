import tensorflow as tf
import ops


def f(x):

    with tf.variable_scope("a", reuse=None):

        x = ops.conv2d(x, 3, [3, 3], [1, 1], "channels_last")

        return x


def g(x):

    with tf.variable_scope("a", reuse=None):

        x = ops.conv2d(x, 3, [3, 3], [1, 1], "channels_last")

        x = ops.conv2d(x, 3, [3, 3], [1, 1], "channels_last", name="xxxxxx")

        return x


with tf.Session() as sess:

    x = tf.constant(1.0, shape=[10, 4, 4, 3])

    with tf.variable_scope("sc", reuse=tf.AUTO_REUSE):

        y = f(x)

    saver = tf.train.Saver()

    checkpoint = tf.train.latest_checkpoint("model")

    if checkpoint:
        saver.restore(sess, checkpoint)
        print(checkpoint, "loaded")

    else:
        sess.run(tf.global_variables_initializer())
        print("global variables initialized")

    saver.save(sess, "model/model.ckpt")

    with tf.variable_scope("sc", reuse=tf.AUTO_REUSE):

        y = g(x)

        uninitialized_variable_names = sess.run(tf.report_uninitialized_variables())
        print(uninitialized_variable_names)
        uninitialized_variables = [var for var in tf.global_variables() if sess.run(tf.is_variable_initialized(var))]

        sess.run(tf.variables_initializer(uninitialized_variables))
        print("uninitialized variables initialized")

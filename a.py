import tensorflow as tf

with tf.variable_scope("scope"):

    x1 = tf.placeholder(dtype=tf.float32, shape=[])
    y1 = tf.get_variable("x", shape=[])
    z1 = x1 * y1
    loss1 = tf.nn.l2_loss(z1)
    train_op1 = tf.train.AdamOptimizer().minimize(loss1)

with tf.variable_scope("scope", reuse=tf.AUTO_REUSE):

    x2 = tf.placeholder(dtype=tf.float32, shape=[])
    y2 = tf.get_variable("x", shape=[])
    z2 = x2 * y2
    loss2 = tf.nn.l2_loss(z2)
    train_op2 = tf.train.AdamOptimizer().minimize(loss2)

with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    print(session.run(train_op2, feed_dict={x2: 10}))
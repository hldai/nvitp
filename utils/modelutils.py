import tensorflow as tf


def apply_linear(x, input_dim, output_dim, scope='linear'):
    with tf.variable_scope(scope):
        W = tf.get_variable("W", dtype=tf.float32, shape=[input_dim, output_dim])
        b = tf.get_variable(
            "b", shape=[output_dim], dtype=tf.float32, initializer=tf.zeros_initializer())
    return tf.matmul(x, W) + b

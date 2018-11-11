import tensorflow as tf
import numpy as np


np.random.seed(123)
W = tf.placeholder(tf.float32, [3, 5], "W")
# W_sm = tf.nn.softmax(W, axis=1)
# W_sm = tf.nn.softmax(W, axis=0)
v = tf.constant([1, 2, 3], tf.float32, shape=[3, 1])
x = v * W

sess = tf.Session()
W_input = np.random.uniform(0, 2, [3, 5])
print(W_input)
x_val = sess.run([x], feed_dict={W: W_input})[0]
print(x_val)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def __train():
    input_dim = 2
    hdim = 5
    odim = 2

    x1 = tf.placeholder(tf.float32, input_dim, 'x1')
    x2 = tf.placeholder(tf.float32, input_dim, 'x2')
    W1 = tf.get_variable('W1', shape=(input_dim, hdim), initializer=tf.contrib.layers.xavier_initializer(),
                         dtype=tf.float32)
    W2 = tf.get_variable('W2', shape=(hdim, odim), initializer=tf.contrib.layers.xavier_initializer(),
                         dtype=tf.float32)

    h1 = tf.matmul(tf.reshape(x1, (-1, input_dim)), W1)
    h2 = tf.matmul(tf.reshape(x2, (-1, input_dim)), W1)

    h1 = tf.nn.tanh(h1)
    h2 = tf.nn.tanh(h2)

    o1 = tf.matmul(h1, W2)
    o2 = tf.matmul(h2, W2)

    loss1 = tf.reduce_sum(tf.square(o1 - o2))
    loss2 = tf.reduce_sum(-tf.square(o1 - o2))

    optimizer = tf.train.AdamOptimizer(0.001)
    train_op1 = optimizer.minimize(loss1)
    train_op2 = optimizer.minimize(loss2)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        losses1 = list()
        losses2 = list()
        for j in range(50):
            feed_dict = {
                x1: data1[j],
                x2: data2[j]
            }
            _, lv1 = sess.run([train_op1, loss1], feed_dict=feed_dict)

            feed_dict = {
                x1: data1[50 + j],
                x2: data2[50 + j]
            }
            _, lv2 = sess.run([train_op2, loss2], feed_dict=feed_dict)
            losses1.append(lv1)
            losses2.append(lv2)
            # print(v1)
            # print(v2)
        print(sum(losses1) / len(losses1), sum(losses2) / len(losses2))

    td1, td2 = list(), list()
    for i in range(100):
        tmp1, tmp2 = sess.run([o1, o2], feed_dict={x1: data1[i], x2: data2[i]})
        td1.append(tmp1[0])
        td2.append(tmp2[0])
    return td1, td2


np.random.seed(19680801)
data1 = np.random.randn(100, 2)
data2 = np.random.randn(100, 2)
td1, td2 = __train()
td1 = np.asarray(td1)
td2 = np.asarray(td2)

# f = plt.figure()

# print(td1[:, 0])
plt.plot(td1[:50, 0], td1[:50, 1], 'b+')
plt.plot(td1[50:, 0], td1[50:, 1], 'bo')
plt.plot(td2[:50, 0], td2[:50, 1], 'r+')
plt.plot(td2[50:, 0], td2[50:, 1], 'ro')
# plt.plot(data[0][50:], data[1][50:], 'ro')
plt.show()

# f.savefig("d:/data/tmp/foo.pdf", bbox_inches='tight')

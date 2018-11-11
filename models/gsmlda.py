import tensorflow as tf
from utils import modelutils
import numpy as np


class GSMLDA:
    def __init__(self, reader, learning_rate=0.001, decay_rate=0.96):
        self.reader = reader
        self.input_dim = self.reader.vocab_size
        self.n_words = self.reader.vocab_size
        self.word_vec_dim = 100
        self.q_hdim = 100
        self.q_outdim = 100
        self.n_topics = 10
        self.sess = None
        self.step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(
            learning_rate, self.step, 10000, decay_rate, staircase=True, name="lr")
        self.__build_model()

    def __build_model(self):
        self.x = tf.placeholder(tf.float32, [self.input_dim], name="input")
        self.x_idx = tf.placeholder(tf.int32, [None], name="x_idx")
        self.word_vecs = tf.get_variable("v", dtype=tf.float32, shape=[self.n_words, self.word_vec_dim])
        self.topic_vecs = tf.get_variable("t", dtype=tf.float32, shape=[self.n_topics, self.word_vec_dim])

        with tf.variable_scope('q-mlp'):
            x_tmp = tf.reshape(self.x, [-1, self.input_dim])
            self.l1_lin = modelutils.apply_linear(x_tmp, self.input_dim, self.q_hdim, scope='l1')
            self.l1 = tf.nn.relu(self.l1_lin)

            self.l2_lin = modelutils.apply_linear(self.l1, self.q_hdim, self.q_outdim, scope='l2')
            self.l2 = tf.nn.relu(self.l2_lin)

            self.mu = modelutils.apply_linear(self.l2, self.q_outdim, self.n_topics, scope='mu')
            self.log_sigma_sq = modelutils.apply_linear(self.l2, self.q_outdim, self.n_topics, scope="log_sigma_sq")

            self.eps = tf.random_normal((1, self.n_topics), 0, 1, dtype=tf.float32)
            self.sigma = tf.sqrt(tf.exp(self.log_sigma_sq))
            self.theta = tf.add(self.mu, tf.multiply(self.sigma, self.eps))
            self.theta = tf.nn.softmax(self.theta)

        # topic word probabilities
        self.beta = tf.nn.softmax(tf.matmul(self.topic_vecs, tf.transpose(self.word_vecs)))

        self.word_probs = tf.matmul(self.theta, self.beta)
        self.log_word_probs = tf.log(self.word_probs)
        self.loss1 = -tf.reduce_sum(self.x * self.log_word_probs)
        self.loss2 = -0.5 * tf.reduce_sum(1 + self.log_sigma_sq - tf.square(self.mu) - tf.exp(self.log_sigma_sq))
        self.loss = self.loss1 + self.loss2

        self.optim = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=self.step)

    def train(self):
        print(self.n_words)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        iterator = self.reader.iterator()
        losses = list()
        for step in range(0, 100000):
            x, x_idx = next(iterator)

            _, loss, mu, sigma, beta_val = self.sess.run(
                [self.optim, self.loss, self.mu, self.sigma, self.beta],
                feed_dict={self.x: x})
            losses.append(loss)

            if step % 1000 == 0:
                print('step={} loss={}'.format(step, sum(losses) / len(losses)))
                self.__show_topics(beta_val)

    def __show_topics(self, beta_val):
        for topic_probs in beta_val:
            idxs = np.argpartition(-topic_probs, range(5))[:5]
            print(' '.join([self.reader.idx2word[idx] for idx in idxs]))

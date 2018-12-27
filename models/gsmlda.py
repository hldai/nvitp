import tensorflow as tf
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from utils import modelutils
from data import bowdataset


class GSMLDA:
    def __init__(self, vocab_size, learning_rate=0.001, decay_rate=0.96):
        # self.reader = reader
        # self.input_dim = self.reader.vocab_size
        # self.n_words = self.reader.vocab_size
        self.vocab_size = vocab_size
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
        self.x = tf.placeholder(tf.float32, [self.vocab_size], name="input")
        self.x_idx = tf.placeholder(tf.int32, [None], name="x_idx")
        self.word_vecs = tf.get_variable("v", dtype=tf.float32, shape=[self.vocab_size, self.word_vec_dim])
        self.topic_vecs = tf.get_variable("t", dtype=tf.float32, shape=[self.n_topics, self.word_vec_dim])

        with tf.variable_scope('q-mlp'):
            x_tmp = tf.reshape(self.x, [-1, self.vocab_size])
            self.l1_lin, self.l1_W, self.l1_b = modelutils.apply_linear(x_tmp, self.vocab_size, self.q_hdim, scope='l1')
            self.l1 = tf.nn.relu(self.l1_lin)

            self.l2_lin, self.l2_W, self.l2_b = modelutils.apply_linear(self.l1, self.q_hdim, self.q_outdim, scope='l2')
            self.l2 = tf.nn.relu(self.l2_lin)

            self.mu, _, _ = modelutils.apply_linear(self.l2, self.q_outdim, self.n_topics, scope='mu')
            self.log_sigma_sq, self.W_lss, _ = modelutils.apply_linear(
                self.l2, self.q_outdim, self.n_topics, scope="log_sigma_sq")

            self.eps = tf.random_normal((1, self.n_topics), 0, 1, dtype=tf.float32)
            # self.sigma = tf.sqrt(tf.exp(self.log_sigma_sq))
            self.sigma = self.log_sigma_sq
            self.theta = tf.add(self.mu, tf.multiply(self.sigma, self.eps))
            self.theta = tf.nn.softmax(self.theta)

        # topic word probabilities
        self.beta = tf.nn.softmax(tf.matmul(self.topic_vecs, tf.transpose(self.word_vecs)))

        self.word_probs = tf.matmul(self.theta, self.beta)
        self.log_word_probs = tf.log(self.word_probs)
        self.loss1 = -tf.reduce_sum(self.x * self.log_word_probs)
        # self.loss2 = -0.5 * tf.reduce_sum(1 + self.log_sigma_sq - tf.square(self.mu) - tf.exp(self.log_sigma_sq))
        self.loss2 = -0.5 * tf.reduce_sum(
            1 + tf.log(tf.square(self.sigma)) - tf.square(self.mu) - tf.square(self.sigma))
        self.loss = self.loss1 + self.loss2

        self.optim = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=self.step)

    def get_best_topic(self, x, beta):
        log_probs = np.log(beta)
        max_log_prob = -1e8
        best_topic_idx = 0
        for i in range(self.n_topics):
            log_prob = np.sum(x * log_probs[i])
            if log_prob > max_log_prob:
                best_topic_idx = i
                max_log_prob = log_prob
        return best_topic_idx, max_log_prob

    def eval(self, dataset, labels, beta):
        labels_sys = list()
        for i in range(dataset.n_examples):
            x = dataset.get_example(i)
            best_topic_idx, max_log_prob = self.get_best_topic(x, beta)
            labels_sys.append(best_topic_idx)
        print(normalized_mutual_info_score(labels, labels_sys))

    def train(self, n_train_steps, train_text_file, train_labels, vocab, idx2word_dict):
        import math
        print(self.vocab_size)

        dataset_train = bowdataset.BowDataset(train_text_file, vocab, idx2word_dict)
        # print(dataset_train.n_examples)
        # print(len(train_labels))
        # exit()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # iterator = self.reader.iterator()
        losses = list()
        rand_idxs = np.random.permutation(np.arange(dataset_train.n_examples))
        for step in range(n_train_steps):
            example_idx = rand_idxs[step % dataset_train.n_examples]
            x = dataset_train.get_example(example_idx)
            # x, x_idx = next(iterator)

            _, loss, beta_val, l1, l2, l3, l4, l5, l6 = self.sess.run(
                [self.optim, self.loss, self.beta, self.l1_lin, self.l2_lin, self.mu, self.log_sigma_sq, self.theta,
                 self.log_word_probs],
                feed_dict={self.x: x})
            # loss, beta_val, l1, l2, l3, l4, l5, l6 = self.sess.run(
            #     [self.loss, self.beta, self.l1_lin, self.l2_lin, self.mu, self.log_sigma_sq, self.theta,
            #      self.log_word_probs],
            #     feed_dict={self.x: x})

            # if math.isnan(loss) or step > 1200:
            if math.isnan(loss):
                tf_trainable_variables = tf.trainable_variables()
                trainable_vals = self.sess.run(tf_trainable_variables, feed_dict={self.x: x})
                x_str = ''
                for i, v in enumerate(x):
                    if v > 0:
                        x_str += '{}:{} '.format(i, v)
                print('step', step)
                print(x_str)
                print('l1')
                print(l1)
                print('l2')
                print(l2)
                print('mu')
                print(l3)
                print('log_sig_sq')
                print(l4)
                print('theta')
                print(l5)
                print('word_probs')
                print(l6)
                print('loss', loss)
                for tv, v in zip(tf_trainable_variables, trainable_vals):
                    print(tv.name)
                    print(v)
                print()
            if math.isnan(loss):
                print('NAN!!')
                exit()
            losses.append(loss)

            # if step > 1370:
            #     break

            if step % 2000 == 0:
                print('step={} loss={}'.format(step, sum(losses) / len(losses)))
                losses = list()
            if step % 2000 == 0:
                self.__show_topics(beta_val, idx2word_dict)
                self.eval(dataset_train, train_labels, beta_val)

    @staticmethod
    def __show_topics(beta_val, idx2word):
        n_topic_words = 10
        for topic_probs in beta_val:
            idxs = np.argpartition(-topic_probs, range(n_topic_words))[:n_topic_words]
            print(' '.join([idx2word[idx] for idx in idxs]))

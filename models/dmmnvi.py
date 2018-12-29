import tensorflow as tf
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from utils import modelutils
from data import bowdataset
from sklearn.cluster import KMeans


class DMMNVI:
    def __init__(self, vocab_size, n_topics, learning_rate=0.001, decay_rate=0.96, lamb_l2_reg=0.0,
                 n_labels_supervised=20):
        # self.reader = reader
        # self.input_dim = self.reader.vocab_size
        # self.n_words = self.reader.vocab_size
        self.vocab_size = vocab_size
        self.word_vec_dim = 100
        self.q_hdim = 100
        self.q_outdim = 100
        self.n_topics = n_topics
        self.lamb_l2_reg = lamb_l2_reg
        self.sess = None
        self.n_labels_supervised = n_labels_supervised
        self.step = tf.Variable(0, trainable=False)
        self.step_supervised = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(
            learning_rate, self.step, 10000, decay_rate, staircase=True, name="lr")
        self.__build_model()

    def __build_model(self):
        self.x = tf.placeholder(tf.float32, [self.vocab_size], name="input")
        self.x_idx = tf.placeholder(tf.int32, [None], name="x_idx")
        self.y_labels_input = tf.placeholder(tf.float32, [self.n_labels_supervised], name='labels_input')
        self.word_vecs = tf.get_variable("word_vecs", dtype=tf.float32, shape=[self.vocab_size, self.word_vec_dim])
        self.topic_vecs = tf.get_variable("topic_vecs", dtype=tf.float32, shape=[self.n_topics, self.word_vec_dim])

        with tf.variable_scope('q-mlp'):
            x_tmp = tf.reshape(self.x, [-1, self.vocab_size])
            self.l1_lin, self.l1_W, self.l1_b = modelutils.apply_linear(x_tmp, self.vocab_size, self.q_hdim, scope='l1')
            self.l1 = tf.nn.tanh(self.l1_lin)

            self.l2_lin, self.l2_W, self.l2_b = modelutils.apply_linear(self.l1, self.q_hdim, self.q_outdim, scope='l2')
            self.l2 = tf.nn.tanh(self.l2_lin)

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

        self.log_word_probs_by_topic = tf.reduce_sum(self.x * tf.log(self.beta), axis=1) + tf.log(self.theta)
        self.loss1 = -tf.reduce_logsumexp(self.log_word_probs_by_topic)
        # self.log_word_probs = tf.log(self.word_probs)
        # self.loss1 = -tf.reduce_sum(self.x * self.log_word_probs)
        # self.loss2 = -0.5 * tf.reduce_sum(1 + self.log_sigma_sq - tf.square(self.mu) - tf.exp(self.log_sigma_sq))
        self.loss2 = -0.5 * tf.reduce_sum(
            1 + tf.log(tf.square(self.sigma)) - tf.square(self.mu) - tf.square(self.sigma))
        self.loss = self.loss1 + self.loss2

        with tf.variable_scope('supervised'):
            # self.label_pred_scores, self.W_s, self.b_s = modelutils.apply_linear(
            #     self.l2, self.q_outdim, self.n_labels_supervised, scope='sup-out')
            self.label_pred_scores, self.W_s, self.b_s = modelutils.apply_linear(
                self.mu, self.n_topics, self.n_labels_supervised, scope='sup-out')
            self.label_pred = tf.nn.softmax(self.label_pred_scores)
        self.loss_supervised = -tf.reduce_sum(self.y_labels_input * tf.log(self.label_pred))

        self.l2_reg = 0
        for v in tf.trainable_variables():
            if 'bias' not in v.name:
                self.l2_reg += tf.nn.l2_loss(v)
        self.loss += self.lamb_l2_reg * self.l2_reg
        self.loss_supervised += self.lamb_l2_reg * self.l2_reg

        self.optim = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=self.step)
        self.optim_supervised = tf.train.AdamOptimizer(
            learning_rate=self.lr).minimize(self.loss_supervised, global_step=self.step_supervised)

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
        print('NMI', normalized_mutual_info_score(labels, labels_sys))

    def eval_k_means(self, dataset, labels):
        vecs = list()
        for i in range(dataset.n_examples):
            x = dataset.get_example(i)
            vec = self.sess.run(self.mu, feed_dict={self.x: x})
            vecs.append(np.squeeze(vec))
        vecs = np.array(vecs, dtype=np.float32)
        print('doing kmeans ...')
        kmeans = KMeans(n_clusters=self.n_topics, random_state=0).fit(vecs)
        print('NMI', normalized_mutual_info_score(labels, kmeans.labels_))

    def train_supervised(self, x, y_true):
        _, loss, lp, lps, wv, bv, l2v, l2w, l2b, l1v, l1w, l1b = self.sess.run(
            [self.optim_supervised, self.loss_supervised, self.label_pred, self.label_pred_scores,
             self.W_s, self.b_s, self.l2, self.l2_W, self.l2_b, self.l1, self.l1_W, self.l1_b],
            feed_dict={self.x: x, self.y_labels_input: y_true})
        return loss, lp

    @staticmethod
    def _get_x_str(x):
        x_str = ''
        for i, v in enumerate(x):
            if v > 0:
                x_str += '{}:{} '.format(i, v)
        return x_str

    def _print_trainable_variable(self, x):
        tf_trainable_variables = tf.trainable_variables()
        trainable_vals = self.sess.run(tf_trainable_variables, feed_dict={self.x: x})
        for tv, v in zip(tf_trainable_variables, trainable_vals):
            print(tv.name)
            print(v)
        print()

    def _print_intermediate_variabels(self, x):
        variables_dict = {'l1_lin': self.l1_lin, 'l2_lin': self.l2_lin, 'mu': self.mu,
                          'beta': self.beta, 'theta': self.theta}
        var_names = [n for n, v in variables_dict.items()]
        tf_vars = [v for n, v in variables_dict.items()]
        vals = self.sess.run(tf_vars, feed_dict={self.x: x})
        for name, v in zip(var_names, vals):
            print(name)
            print(v)

    def train(self, n_train_steps, train_text_file, train_labels, vocab, idx2word_dict):
        import math
        print(self.vocab_size)

        dataset_train = bowdataset.BowDataset(train_text_file, vocab, idx2word_dict)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # iterator = self.reader.iterator()
        losses, losses_supervised = list(), list()
        rand_idxs = np.random.permutation(np.arange(dataset_train.n_examples))
        labels_true, labels_pred = list(), list()
        for step in range(n_train_steps):
            example_idx = rand_idxs[step % dataset_train.n_examples]
            x = dataset_train.get_example(example_idx)

            y_true = np.zeros(self.n_labels_supervised, np.float32)
            y_true[train_labels[example_idx]] = 1
            loss_supervised, y_pred = self.train_supervised(x, y_true)
            # print(y_true, y_pred)
            # print(train_labels[example_idx], np.argmax(np.squeeze(y_pred)))
            labels_true.append(train_labels[example_idx])
            labels_pred.append(np.argmax(np.squeeze(y_pred)))
            losses_supervised.append(loss_supervised)
            if math.isnan(loss_supervised):
                break

            # if step % 100 == 0:
            #     self._print_intermediate_variabels(x)

            _, loss, beta_val = self.sess.run(
                [self.optim, self.loss, self.beta], feed_dict={self.x: x})
            # loss, beta_val, l1, l2, l3, l4, l5, l6 = self.sess.run(
            #     [self.loss, self.beta, self.l1_lin, self.l2_lin, self.mu, self.log_sigma_sq, self.theta,
            #      self.log_word_probs],
            #     feed_dict={self.x: x})

            # if math.isnan(loss) or step % 100 == 0:
            if math.isnan(loss):
                print('step', step)
                x_str = self._get_x_str(x)
                print(x_str)
                self._print_trainable_variable(x)
            if math.isnan(loss):
                print('NAN!!')
                exit()
            losses.append(loss)

            if step % 2000 == 0:
                print('step={} loss={} loss_l={}'.format(
                    step, sum(losses), sum(losses_supervised)))
                losses, losses_supervised = list(), list()
                if labels_pred:
                    print('acc={}'.format(accuracy_score(labels_true, labels_pred)))
                    labels_true, labels_pred = list(), list()
            # if step % 2000 == 0:
            #     self.__show_topics(beta_val, idx2word_dict)
            #     self.eval(dataset_train, train_labels, beta_val)
            if (step + 1) % 2000 == 0:
                self.eval_k_means(dataset_train, train_labels)

    @staticmethod
    def __show_topics(beta_val, idx2word):
        n_topic_words = 10
        for topic_probs in beta_val:
            idxs = np.argpartition(-topic_probs, range(n_topic_words))[:n_topic_words]
            print(' '.join([idx2word[idx] for idx in idxs]))

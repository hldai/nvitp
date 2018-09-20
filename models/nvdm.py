import tensorflow as tf
import numpy as np
import os
import time


class NVDM:
    def __init__(self, reader, dataset, decay_rate=0.96, decay_step=10000, embed_dim=100,
                 h_dim=50, learning_rate=0.001, max_iter=450000, checkpoint_dir="checkpoint"):
        self.reader = reader
        self.saver = None
        self.sess = None

        self.h_dim = h_dim
        self.embed_dim = embed_dim

        self.max_iter = max_iter
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.checkpoint_dir = checkpoint_dir
        self.step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(
            learning_rate, self.step, 10000, decay_rate, staircase=True, name="lr")

        # _ = tf.scalar_summary("learning rate", self.lr)

        self.dataset = dataset
        self._attrs = ["h_dim", "embed_dim", "max_iter", "dataset",
                       "learning_rate", "decay_rate", "decay_step"]

        self.__build_model()

    def __build_model(self):
        self.input_dim = self.reader.vocab_size
        print('input dim={}'.format(self.input_dim))
        self.x = tf.placeholder(tf.float32, [self.input_dim], name="input")
        self.x_idx = tf.placeholder(tf.int32, [None], name="x_idx")

        self.__build_encoder()
        self.__build_generator()

        # Kullback Leibler divergence
        self.e_loss = -0.5 * tf.reduce_sum(1 + self.log_sigma_sq - tf.square(self.mu) - tf.exp(self.log_sigma_sq))

        # Log likelihood
        self.g_loss = -tf.reduce_sum(tf.log(tf.gather(self.p_x_i, self.x_idx) + 1e-10))

        self.loss = self.e_loss + self.g_loss

        self.encoder_var_list, self.generator_var_list = [], []
        for var in tf.trainable_variables():
            if "encoder" in var.name:
                self.encoder_var_list.append(var)
            elif "generator" in var.name:
                self.generator_var_list.append(var)

        # optimizer for alternative update
        self.optim_e = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
            self.e_loss, global_step=self.step, var_list=self.encoder_var_list)
        self.optim_g = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
            self.g_loss, global_step=self.step, var_list=self.generator_var_list)

        # optimizer for one shot update
        self.optim = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=self.step)

        # _ = tf.scalar_summary("encoder loss", self.e_loss)
        # _ = tf.scalar_summary("generator loss", self.g_loss)
        # _ = tf.scalar_summary("total loss", self.loss)

    def __build_encoder(self):
        with tf.variable_scope('encoder'):
            x_tmp = tf.reshape(self.x, [-1, self.input_dim])
            self.l1_lin = NVDM.apply_linear(x_tmp, self.input_dim, self.embed_dim, scope='l1')
            self.l1 = tf.nn.relu(self.l1_lin)

            self.l2_lin = NVDM.apply_linear(self.l1, self.embed_dim, self.embed_dim, scope='l2')
            self.l2 = tf.nn.relu(self.l2_lin)

            self.mu = NVDM.apply_linear(self.l2, self.embed_dim, self.h_dim, scope='mu')
            self.log_sigma_sq = NVDM.apply_linear(self.l2, self.embed_dim, self.h_dim, scope="log_sigma_sq")

            self.eps = tf.random_normal((1, self.h_dim), 0, 1, dtype=tf.float32)
            self.sigma = tf.sqrt(tf.exp(self.log_sigma_sq))
            self.h = tf.add(self.mu, tf.multiply(self.sigma, self.eps))

    def __build_generator(self):
        with tf.variable_scope('generator'):
            self.R = tf.get_variable('R', [self.h_dim, self.input_dim])
            self.b_gen = tf.get_variable('b', [self.input_dim])

            self.e = -tf.matmul(self.h, self.R) + self.b_gen
            self.p_x_i = tf.squeeze(tf.nn.softmax(self.e))

    @staticmethod
    def apply_linear(x, input_dim, output_dim, scope='linear'):
        with tf.variable_scope(scope):
            W = tf.get_variable("W", dtype=tf.float32, shape=[input_dim, output_dim])
            b = tf.get_variable(
                "b", shape=[output_dim], dtype=tf.float32, initializer=tf.zeros_initializer())
        return tf.matmul(x, W) + b

    def train(self):
        self.sess = tf.Session()
        # self.load(self.checkpoint_dir)
        self.sess.run(tf.global_variables_initializer())

        start_time = time.time()
        start_iter = self.step.eval(session=self.sess)
        iterator = self.reader.iterator()
        for step in range(start_iter, start_iter + self.max_iter):
            x, x_idx = next(iterator)

            _, loss, mu, sigma, h = self.sess.run(
                [self.optim, self.loss, self.mu, self.sigma, self.h],
                feed_dict={self.x: x, self.x_idx: x_idx})

            if step % 1000 == 0:
                print("Step: [%4d/%4d] time: %4.4f, loss: %.8f" % (
                    step, self.max_iter, time.time() - start_time, loss))

    def load(self, checkpoint_dir):
        self.saver = tf.train.Saver()

        print(" [*] Loading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Load SUCCESS")
            return True
        else:
            print(" [!] Load failed...")
            return False

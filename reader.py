import os
import itertools
import numpy as np
import tensorflow as tf

from utils import utils
from collections import Counter
from nltk.tokenize import TreebankWordTokenizer

EOS_TOKEN = "_eos_"


class TextReader(object):
    def __init__(self, data_path):
        train_path = os.path.join(data_path, "train.txt")
        valid_path = os.path.join(data_path, "valid.txt")
        test_path = os.path.join(data_path, "test.txt")
        vocab_path = os.path.join(data_path, "vocab.pkl")

        if os.path.exists(vocab_path):
            self._load(vocab_path, train_path, valid_path, test_path)
        else:
            self._build_vocab(train_path, vocab_path)
            self.train_data = self._file_to_data(train_path)
            self.valid_data = self._file_to_data(valid_path)
            self.test_data = self._file_to_data(test_path)

        self.idx2word = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def onehot(self, data, min_length=None):
        if min_length is None:
            min_length = self.vocab_size
        return np.bincount(data, minlength=min_length)

    def get_data_from_type(self, data_type):
        if data_type == "train":
            raw_data = self.train_data
        elif data_type == "valid":
            raw_data = self.valid_data
        elif data_type == "test":
            raw_data = self.test_data
        else:
            raise Exception(" [!] Unkown data type %s: %s" % data_type)
        return raw_data

    def iterator(self, data_type='train'):
        raw_data = self.get_data_from_type(data_type)

        # citer = itertools.cycle([d for d in raw_data if d])
        # print(next(citer))
        return itertools.cycle([[self.onehot(data), data] for data in raw_data if len(data) > 0])

    def _load(self, vocab_path, train_path, valid_path, test_path):
        self.vocab = utils.load_pkl(vocab_path)
        self.train_data = utils.load_npy(train_path + ".npy")
        self.valid_data = utils.load_npy(valid_path + ".npy")
        self.test_data = utils.load_npy(test_path + ".npy")

    @staticmethod
    def _read_text(file_path):
        with open(file_path, encoding='utf-8') as f:
            return f.read().replace('\n', ' %s ' % EOS_TOKEN)

    def _build_vocab(self, file_path, vocab_path):
        counter = Counter(TextReader._read_text(file_path).split(' '))
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
        self.vocab = dict(zip(words, range(len(words))))
        utils.save_pkl(vocab_path, self.vocab)

    def _file_to_data(self, file_path):
        texts = self._read_text(file_path).split(EOS_TOKEN)
        data = list()
        for text in texts:
            v = np.array(list(map(self.vocab.get, text.split())))
            data.append(v)

        utils.save_npy(file_path + '.npy', data)
        return data

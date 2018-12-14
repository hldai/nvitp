import numpy as np


class BowDataset:
    def __init__(self, text_file, vocab, idx2word_dict):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.idx2word_dict = idx2word_dict
        self.word_idxs_list = list()
        self.word_cnts_list = list()
        self.__load_texts(text_file)
        self.n_examples = len(self.word_idxs_list)

    def get_example(self, example_idx):
        word_idxs = self.word_idxs_list[example_idx]
        word_cnts = self.word_cnts_list[example_idx]
        x = np.zeros(self.vocab_size, np.int32)
        for idx, cnt in zip(word_idxs, word_cnts):
            x[idx] = cnt
        return x

    def __load_texts(self, text_file):
        print('loading {} ...'.format(text_file))
        f = open(text_file, encoding='utf-8')
        for line in f:
            words = line.strip().split(' ')
            word_idx_cnt_tups = self.__get_word_cnt_tups(words, self.vocab)
            if not word_idx_cnt_tups:
                continue
            word_idxs = [idx for idx, _ in word_idx_cnt_tups]
            word_cnts = [cnt for _, cnt in word_idx_cnt_tups]
            self.word_idxs_list.append(word_idxs)
            self.word_cnts_list.append(word_cnts)
        f.close()
        print('{} docs loaded'.format(len(self.word_cnts_list)))

    @staticmethod
    def __get_word_cnt_tups(words, vocab):
        word_idx_cnt_dict = dict()
        for w in words:
            idx = vocab.get(w, -1)
            if idx < 0:
                continue
            cnt = word_idx_cnt_dict.get(idx, 0)
            word_idx_cnt_dict[idx] = cnt + 1
        word_idx_cnt_tups = list(word_idx_cnt_dict.items())
        word_idx_cnt_tups.sort(key=lambda x: x[0])
        return word_idx_cnt_tups

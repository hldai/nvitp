import numpy as np
import os
import tensorflow as tf
import config
from utils import datautils, utils
from data import bowdataset
from models.nvdm import NVDM
from models.gsmlda import GSMLDA
from models.dmmnvi import DMMNVI


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    n_topics = 20
    min_word_cnt = 20
    max_word_cnt = 1000
    n_train_steps = 150000
    remove_stopwords = True
    # dataset_files = config.PTB_FILES
    dataset_files = config.TNG_FILES

    # reader = TextReader(data_path)
    # model = NVDM(reader, dataset)
    # model.train()

    vocab = utils.load_vocab(dataset_files['word_cnt_file'], max_word_cnt, min_word_cnt,
                             min_word_len=3, remove_stopwords=remove_stopwords)
    idx2word_dict = {idx: w for w, idx in vocab.items()}
    # dataset = bowdataset.BowDataset(dataset_files['train_text_file'], vocab, idx2word_dict)
    print(len(vocab), 'words in vocab')

    train_labels = datautils.load_tng_labels(dataset_files['train_labels_file'])

    tf.random.set_random_seed(127)
    # model = GSMLDA(len(vocab))
    model = DMMNVI(len(vocab), n_topics=n_topics)
    model.train(n_train_steps, dataset_files['train_tok_text_file'], train_labels, vocab, idx2word_dict)

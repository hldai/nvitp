import numpy as np
import config
from utils import datautils, utils
from data import bowdataset
from models.nvdm import NVDM
from models.gsmlda import GSMLDA


if __name__ == '__main__':
    dataset = 'ptb'
    min_word_cnt = 10
    max_word_cnt = 1000
    # dataset_files = config.PTB_FILES
    dataset_files = config.TNG_FILES

    # reader = TextReader(data_path)
    # model = NVDM(reader, dataset)
    # model.train()

    vocab = utils.load_vocab(dataset_files['word_cnt_file'], max_word_cnt, min_word_cnt,
                             min_word_len=3)
    idx2word_dict = {idx: w for w, idx in vocab.items()}
    # dataset = bowdataset.BowDataset(dataset_files['train_text_file'], vocab, idx2word_dict)
    print(len(vocab), 'words in vocab')
    model = GSMLDA(len(vocab))
    model.train(dataset_files['train_tok_text_file'], vocab, idx2word_dict)

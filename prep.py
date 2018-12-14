import config
import os
import pandas as pd
from data import dataprep
from utils import utils, datautils


min_word_cnt = 2
max_word_cnt = 1500
filter_non_alphabet = True
filter_one_time_words = True
# dataset_files = config.PTB_FILES
dataset_files = config.TNG_FILES
# dataprep.merge_20ng(
#     dataset_files['train_docs_dir'], dataset_files['train_text_file'], dataset_files['train_labels_file'])
datautils.gen_word_cnt_file(
    dataset_files['train_tok_text_file'], dataset_files['word_cnt_file'],
    filter_non_alphabet=filter_non_alphabet, filter_one_time=filter_one_time_words)

# datautils.gen_word_cnt_file(dataset_files['train_text_file'], dataset_files['word_cnt_file'])
# datautils.gen_tfrec_file(
#     dataset_files['train_text_file'], dataset_files['word_cnt_file'], min_word_cnt,
#     max_word_cnt, dataset_files['train_tfrec_file'])

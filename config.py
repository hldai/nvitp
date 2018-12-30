from os.path import join
from platform import platform

env = 'Windows' if platform().startswith('Windows') else 'Linux'

if env == 'Windows':
    PTB_DIR = 'd:/data/cdcr/nvitp/ptb'
    TNG_DIR = 'd:/data/20ng'
else:
    PTB_DIR = '/home/hldai/data/cdcr/nvitp/ptb'
    TNG_DIR = '/home/hldai/data/20ng'

TNG_TRAIN_DIR = join(TNG_DIR, '20news-bydate-train')
TNG_TRAIN_TEXTS_FILE = join(TNG_DIR, 'tng-texts-train.txt')
TNG_TRAIN_LABEL_FILE = join(TNG_DIR, 'tng-labels-train.txt')

TNG_FILES = {
    'train_docs_dir': join(TNG_DIR, '20news-bydate-train'),
    'train_text_file': join(TNG_DIR, 'tng-texts-train.txt'),
    'train_tok_text_file': join(TNG_DIR, 'tng-tok-texts-train.txt'),
    'train_labels_file': join(TNG_DIR, 'tng-labels-train.txt'),
    'word_cnt_file': join(TNG_DIR, 'tng-word-cnt.txt')
}

PTB_FILES = {
    'train_text_file': join(PTB_DIR, 'train.txt.'),
    'valid_text_file': join(PTB_DIR, 'valid.txt'),
    'test_text_file': join(PTB_DIR, 'test.txt'),
    'train_tfrec_file': join(PTB_DIR, 'train.tfrecord'),
    'valid_tfrec_file': join(PTB_DIR, 'valid.tfrecord'),
    'test_tfrec_file': join(PTB_DIR, 'test.tfrecord'),
    'word_cnt_file': join(PTB_DIR, 'word-cnt.txt'),
}

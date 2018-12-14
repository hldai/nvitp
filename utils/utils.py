import pprint
import pickle
import numpy as np

pp = pprint.PrettyPrinter()


def save_pkl(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        print(" [*] save %s" % path)


def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
        print(" [*] load %s" % path)
        return obj


def save_npy(path, obj):
    np.save(path, obj)
    print(" [*] save %s" % path)


def load_npy(path):
    obj = np.load(path)
    print(" [*] load %s" % path)
    return obj


def load_vocab(word_cnt_file, max_word_cnt, min_word_cnt, min_word_len=2, max_word_len=-1):
    import pandas as pd
    df = pd.read_csv(word_cnt_file, na_filter=False)
    vocab = dict()
    widx = 0
    for w, cnt in df.itertuples(False, None):
        if len(w) < min_word_len:
            continue
        if min_word_cnt <= cnt <= max_word_cnt:
            vocab[w] = widx
            widx += 1
    return vocab

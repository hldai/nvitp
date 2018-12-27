import pandas as pd
import tensorflow as tf
import collections
from utils import utils


def load_tng_labels(tng_labels_file):
    with open(tng_labels_file, encoding='utf-8') as f:
        df = pd.read_csv(f)
    return df['il'].as_matrix()


def __has_alphabet(w: str):
    for ch in w:
        if ch.isalpha():
            return True
    return False


def gen_word_cnt_file(filename, output_file, filter_non_alphabet=False, filter_one_time=True):
    print('gen vocab for', filename)
    print('to', output_file)
    word_cnt_dict = dict()
    f = open(filename, encoding='utf-8')
    for line in f:
        words = set(line.strip().split(' '))
        for w in words:
            if filter_non_alphabet and not __has_alphabet(w):
                continue
            cnt = word_cnt_dict.get(w, 0)
            word_cnt_dict[w] = cnt + 1
    f.close()

    word_cnt_tups = [(w, cnt) for w, cnt in word_cnt_dict.items() if cnt > 1]
    word_cnt_tups.sort(key=lambda x: -x[1])
    with open(output_file, 'w', encoding='utf-8', newline='\n') as fout:
        pd.DataFrame(word_cnt_tups, columns=['word', 'cnt']).to_csv(fout, index=False)


def __get_tfrec_feature_dict(word_idx_cnt_tups):
    word_idxs = [idx for idx, _ in word_idx_cnt_tups]
    word_cnts = [cnt for _, cnt in word_idx_cnt_tups]

    def create_int_feature(values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    features = collections.OrderedDict()
    features["word_ids"] = create_int_feature(word_idxs)
    features["word_cnts"] = create_int_feature(word_cnts)
    return features


def gen_tfrec_file(text_file, word_cnt_file, min_word_cnt, max_word_cnt, output_file):
    vocab = utils.load_vocab(word_cnt_file, max_word_cnt, min_word_cnt)
    f = open(text_file, encoding='utf-8')
    writer = tf.python_io.TFRecordWriter(output_file)
    for i, line in enumerate(f):
        words = line.strip().split(' ')
        word_idx_cnt_tups = __get_word_cnt_tups(words, vocab)
        tf_features = __get_tfrec_feature_dict(word_idx_cnt_tups)
        tf_example = tf.train.Example(features=tf.train.Features(feature=tf_features))
        writer.write(tf_example.SerializeToString())
    f.close()
    writer.close()


def get_dataset(data_file, batch_size, is_train):
    dataset = tf.data.TFRecordDataset(data_file)
    name_to_features = {
        "word_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "word_cnts": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)

        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    if is_train:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=100)

    drop_remainder = True if is_train else False
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return dataset

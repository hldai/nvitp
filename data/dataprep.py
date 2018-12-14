import pandas as pd
import os
import re


def __get_beg_pos(text):
    p = 0
    while p < len(text) and text[p] in {'>', '|', ' ', ':', 'X', '#'}:
        p += 1
    return p


def __get_20ng_text(filename):
    text = ''
    # f = open(filename, encoding='utf-8')
    f = open(filename, encoding='windows-1252')
    for line in f:
        if not line.strip():
            break
    for line in f:
        beg_pos = __get_beg_pos(line)
        if beg_pos >= len(line):
            continue
        text += line[beg_pos:]
    f.close()
    return text


def merge_20ng(docs_dir, texts_output_file, labels_output_file):
    label_tups = list()
    fout_text = open(texts_output_file, 'w', encoding='utf-8')
    for i, folder_name in enumerate(os.listdir(docs_dir)):
        data_path = os.path.join(docs_dir, folder_name)
        assert not os.path.isfile(data_path)
        for filename in os.listdir(data_path):
            file_path = os.path.join(data_path, filename)
            label_tups.append((file_path, folder_name, i))

            doc_text = __get_20ng_text(file_path)
            doc_text = re.sub('\s+', ' ', doc_text)
            fout_text.write('{}\n'.format(doc_text))
            # print(file_path)
            # print(doc_text)
            # break
    fout_text.close()

    df_label = pd.DataFrame(label_tups, columns=['path', 'tl', 'il'])
    with open(labels_output_file, 'w', encoding='utf-8', newline='\n') as fout:
        df_label.to_csv(fout, index=False)

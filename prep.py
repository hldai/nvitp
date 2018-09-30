import config
import os
import pandas as pd


def __get_20ng_text(filename):
    text = ''
    f = open(filename, encoding='utf-8')
    for line in f:
        if not line.strip():
            break
    for line in f:
        text += line
    f.close()
    return text


def __merge_20ng():
    label_tups = list()
    fout_text = open(config.TNG_TRAIN_TEXTS_FILE, 'w', encoding='utf-8')
    for i, folder_name in enumerate(os.listdir(config.TNG_TRAIN_DIR)):
        data_path = os.path.join(config.TNG_TRAIN_DIR, folder_name)
        assert not os.path.isfile(data_path)
        for filename in os.listdir(data_path):
            file_path = os.path.join(data_path, filename)
            label_tups.append((file_path, folder_name, i))

            print(file_path)
            doc_text = __get_20ng_text(file_path)
            print(doc_text)
            break
    fout_text.close()

    df_label = pd.DataFrame(label_tups, columns=['path', 'tl', 'il'])
    with open(config.TNG_TRAIN_LABEL_FILE, 'w', encoding='utf-8', newline='\n') as fout:
        df_label.to_csv(fout, index=False)


__merge_20ng()

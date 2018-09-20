import config
import os
import pandas as pd


def __merge_20ng():
    label_tups = list()
    for i, folder_name in enumerate(os.listdir(config.TNG_TRAIN_DIR)):
        data_path = os.path.join(config.TNG_TRAIN_DIR, folder_name)
        assert not os.path.isfile(data_path)
        for filename in os.listdir(data_path):
            file_path = os.path.join(data_path, filename)
            label_tups.append((file_path, folder_name, i))

    df_label = pd.DataFrame(label_tups, columns=['path', 'tl', 'il'])
    with open(config.TNG_TRAIN_LABEL_FILE, 'w', encoding='utf-8', newline='\n') as fout:
        df_label.to_csv(fout, index=False)


__merge_20ng()

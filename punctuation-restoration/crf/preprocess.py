import os
import pandas as pd
import numpy as np
import re
from nltk.tokenize import regexp
import argparse
from sklearn.model_selection import train_test_split

# parser = argparse.ArgumentParser(description='Process dataframe data.')
#
# parser.add_argument('--text_path',
#                     help='input files', default='../texts/tedtalk2012/')
#
# parser.add_argument('--output_path',
#                     help='Dir to save output files', default='./data/tedtalk2012/')

# args = parser.parse_args()
#
# BASE_DIR = args.text_path
# PATH_TO_SAVE = args.output_path


def split_df(df_):
    sents_ids_uniq = df_.sentence_id.astype(np.int32).unique()
    # TEDTALK proproportion
    trainIdx, testIdx = train_test_split(
        sents_ids_uniq, test_size=0.017593607011664625, random_state=42)

    testIdx, devIdx = train_test_split(
        testIdx, test_size=0.36100936100936104, random_state=42)

    sents_id = df_['sentence_id'].astype(np.int32).to_numpy()

    train_df = df_[np.isin(sents_id, trainIdx)]
    test_df = df_[np.isin(sents_id, testIdx)]
    dev_df = df_[np.isin(sents_id, devIdx)]

    return train_df, dev_df, test_df


def replace(sent_id, sentence):
    tokenizer = regexp.RegexpTokenizer(r'\w+|[.,?]')

    # we lowercasedthe entire corpus with the purpose of eliminating bias
    # around the prediction ofperiods.
    # Automatic punctuation restoration with BERT models #Nagy et. al

    tokens = tokenizer.tokenize(sentence.lower())
    sent_data = []
    for _, token in enumerate(tokens):
        try:
            if token not in ['.', ',', '?']:
                sent_data.append([sent_id, 'O', token])
            elif token == '.':
                sent_data[-1][1] = 'I-PERIOD'

            elif token == ',':
                sent_data[-1][1] = 'I-COMMA'

            elif token == '?':
                sent_data[-1][1] = 'I-QUESTION'
        except IndexError:
            continue

    return sent_data


def preprocess(base_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(base_dir):
        dataset2 = []

        if filename.endswith("train.txt"):
            filetype = "train"
        elif filename.endswith("test.txt"):
            filetype = "test"
        elif filename.endswith("dev.txt"):
            filetype = "dev"
        else:
            filetype = "other"

        if not os.path.isfile(os.path.join(base_dir, filename)):
            continue

        file = open(os.path.join(base_dir, filename))
        data = file.readlines()

        for i, line in enumerate(data):
            text = re.sub(r'[!;]', '.', line)
            text = re.sub(r'[:]', ',', text)
            text = re.sub(r'\s[-]\s', ',', text).lower()

            emotions = re.findall(r'\(\w+\)', text)
            if len(emotions) > 0:
                continue

            dataset2.extend(replace(i, text))

        df = pd.DataFrame(np.array(dataset2), columns=['sentence_id', 'labels', 'words'])
        df.to_csv(os.path.join(output_dir, f'{filetype}.csv'), index=False, index_label=False)


if __name__ == '__main__':
    pass

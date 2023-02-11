import argparse

from datasets import load_dataset
from nltk.tokenize import wordpunct_tokenize
import string
import os

id2label = {
    0: 'O',
    1: 'I-PERIOD',
    2: 'I-COMMA'
}


def tokens2labels(tokens, return_labels=True):
    """
    Convert text to labels
    :param return_labels: return labels or ids
    :param tokens:  list of tokens
    :return:  list of labels
    """

    labels = []
    for i, token in enumerate(tokens):
        try:
            if token not in string.punctuation:
                labels.append(0)
            elif token in ['.', '?', '!', ';']:
                labels[-1] = 1
            elif token == ',':
                labels[-1] = 2

        except IndexError:
            raise ValueError(f"Sentence can't start with punctuation {token}")
    if return_labels:
        labels = list(map(lambda item: id2label[item], labels))
    return labels


def remove_punctuation(text):
    """
    Remove punctuation from text
    :param text: text to remove punctuation from
    :return:  text without punctuation
    """
    text = [word.lower() for word in wordpunct_tokenize(text)
            if word not in string.punctuation]
    return text


def save_dataset(dataset, save_path):
    """
    Save dataset to file
    :param dataset: dataset to save
    :param save_path: path to save dataset
    :return: None
    """
    with open(save_path, 'w') as f:
        for sentence in dataset:
            tokens = wordpunct_tokenize(sentence.lower())
            labels = tokens2labels(tokens)
            tokens = [token for token in tokens if token not in string.punctuation]
            for word, label in zip(tokens, labels):
                f.write(f"{word} {label} \n")
            f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='tiagoblima/punctuation-nilc-bert')
    parser.add_argument('--save_path', type=str, default='dataset')
    parser.add_argument('--split', type=str, default='train, validation, test')
    parser.add_argument('--save_format', type=str, default='txt')
    parser.add_argument('--text_column', type=str, default='sent_text')

    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    dataset = load_dataset(args.dataset_path)
    for split in ['train', 'validation', 'test']:

        split = 'dev' if split == 'validation' else split

        save_dataset(dataset['text'], os.path.join(args.save_path, f'{split}.{args.save_format}'))


if __name__ == '__main__':
    main()

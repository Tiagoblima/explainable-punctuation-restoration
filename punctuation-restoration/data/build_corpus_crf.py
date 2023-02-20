import os
import string

import click
import spacy
from datasets import load_dataset
from nltk.tokenize import wordpunct_tokenize

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


def save_dataset(dataset, save_path, nlp, data_format='conll'):
    """
    Save dataset to file
    :param data_format:
    :param nlp:
    :param dataset: dataset to save
    :param save_path: path to save dataset
    :return: None
    """

    with open(save_path, 'w') as f:
        for sentence in nlp.pipe(dataset, batch_size=1000):
            tokens = [token.text.lower() for token in sentence]
            labels = tokens2labels(tokens)
            tokens_postag = [(token.text.lower(), nlp.vocab[token.pos].text) for token in sentence if
                             not token.is_punct]
            if data_format == 'conll':
                for (word, pos), label in zip(tokens_postag, labels):
                    try:
                        f.write(f"{word} {pos} {label}\n")
                    except UnicodeEncodeError:
                        continue
                f.write('\n')
            elif data_format == 'csv':
                f.write('word,pos,label\n')
                for (word, pos), label in zip(tokens_postag, labels):
                    try:
                        f.write(f"{word},{pos},{label}\n")
                    except UnicodeEncodeError:
                        continue
                f.write('\n')
    print(f"Saved dataset to {save_path}")
    print(f"Dataset size: {len(dataset)}")


@click.command()
@click.option('--dataset_name', type=str, default='tiagoblima/punctuation-nilc-bert')
@click.option('--save_path', type=str, default='datasets')
@click.option('--splits', type=str, default='train,validation,test')
@click.option('--save_format', type=str, default='conll')
@click.option('--text_column', type=str, default='text')
def main(
        dataset_name,
        save_path,
        splits,
        save_format,
        text_column
):
    """
    Build dataset from dataset path
    :param dataset_name:  path to dataset
    :param save_path:  path to save dataset
    :param splits:  split to use
    :param save_format:  format to save dataset
    :param text_column:    column to use as text
    :return:  None
    """

    save_path = os.path.join(save_path, dataset_name)

    os.makedirs(save_path, exist_ok=True)
    nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner', 'lemmatizer', 'textcat'])
    dataset = load_dataset(dataset_name)
    for split in splits.split(','):
        split = split.strip()
        dataset_split = dataset[split]
        split = split.replace('validation', 'dev')
        save_dataset(dataset_split[text_column], os.path.join(save_path, f'{split}.{save_format}'), nlp, save_format)


if __name__ == '__main__':
    main()

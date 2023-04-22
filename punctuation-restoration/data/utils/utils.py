from warnings import WarningMessage
import re
from nltk.tokenize import regexp
from nltk.tokenize import sent_tokenize, wordpunct_tokenize
import string
from transformers import pipeline

##Legend 0 = Other, 1 = I-PERIOD, 2 = I-COMMA
id2label = {
    0: 'O',
    1: 'I-PERIOD',
    2: 'I-COMMA'
}


def text2t5labels(sentence):
    """
    Convert text to labels
    :param sentence: text to convert
    :return:  list of labels
    """
    ref_tokens = wordpunct_tokenize(sentence.lower())

    labels = []
    new_tokens = []
    for i, token in enumerate(ref_tokens):
        try:

            if token not in string.punctuation:
                labels.append(token)
                new_tokens.append(token)

            elif token in ['.', '?', '!', ';']:
                if len(labels) > 1:
                    labels.insert(len(labels) - 1, '[Other]')
                labels.append("[I-PERIOD]")

            elif token == ',':
                if len(labels) > 1:
                    labels.insert(len(labels) - 1, '[Other]')
                labels.append("[I-COMMA]")
        except IndexError:
            raise ValueError(f"Sentence can't start with punctuation {token}")

    return {"text_input": ' '.join(new_tokens), "labels": ' '.join(labels)}



def bert_transform_sentences(text_, groups):
    new_text_list = list(text_)

    shift = 0

    for out in groups:
        punkt = '.' if out['entity_group'] == 'PERIOD' else ','
        if out['end'] + shift < len(new_text_list) + 1:
            new_text_list.insert(out['end'] + shift, punkt)
            shift += 1

    return ''.join(new_text_list)


def get_bert_pred_sentence(sentence: str, model: pipeline):
    groups = model(sentence)

    new_text = bert_transform_sentences(sentence, groups)
    return new_text
def remove_punctuation(text):
    """
    Remove punctuation from text
    :param text: text to remove punctuation from
    :return:  text without punctuation
    """
    text = [word.lower() for word in wordpunct_tokenize(text)
            if word not in string.punctuation]
    return text


def text2labels(sentence):
    """
    Convert text to labels
    :param sentence: text to convert
    :return:  list of labels
    """
    tokens = wordpunct_tokenize(sentence.lower())

    labels = []
    new_tokens = []
    for i, token in enumerate(tokens):
        try:
            if token not in string.punctuation:
                labels.append("O")
                new_tokens.append(token)
            elif token in ['.', '?', '!', ';']:
                labels[-1] = "I-PERIOD"
            elif token == ',':
                labels[-1] = "I-COMMA"

        except IndexError:
            continue
            # raise ValueError(f"Sentence can't start with punctuation {token}")

            # Warning(f"Sentence can't start with punctuation {token}")
    return {"tokens": new_tokens, "labels": labels}

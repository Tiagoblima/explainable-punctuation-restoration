from warnings import WarningMessage
import re
from nltk.tokenize import regexp
from nltk.tokenize import sent_tokenize, wordpunct_tokenize
import string

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
            print(sentence)
            # Warning(f"Sentence can't start with punctuation {token}")
    return {"tokens": new_tokens, "labels": labels}

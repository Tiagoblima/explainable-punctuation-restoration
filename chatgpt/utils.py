import time

import openai
from seqeval.metrics import classification_report
import string
from nltk.tokenize import wordpunct_tokenize
from chatgpt.constants import API_KEY


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
    for i, token in enumerate(tokens):
        try:
            if token not in string.punctuation:
                labels.append('O')
            elif token in ['.', '?', '!', ';']:
                labels[-1] = 'I-PERIOD'
            elif token == ',':
                labels[-1] = 'I-COMMA'

        except IndexError:

            print(f"Sentence can't start with punctuation {token}")
            continue
    return labels


def compute_scores(true_labels, pred_labels):
    new_true_labels = []
    new_pred_labels = []
    for t_lbls, p_lbls in zip(true_labels, pred_labels):
        new_true_labels.append([
            t_lbl for t_lbl, p_lbl in zip(t_lbls, p_lbls)
        ])

        new_pred_labels.append([
            p_lbl for t_lbl, p_lbl in zip(t_lbls, p_lbls)
        ])

    return classification_report(new_true_labels, new_pred_labels, output_dict=True)


openai.api_key = API_KEY


def chat_gpt_predict(messages):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
        except Exception as e:
            time.sleep(20)
            continue
        break
    pred_text = response.choices[0].message.content.replace("\"", "")
    return pred_text

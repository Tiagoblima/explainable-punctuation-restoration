import re

from data_annotation.preprocess_data import special_pattern, marks


def join_punctuation_marks(text):
    text = re.sub(r'(\w)\s([.,?!;:]+)', r'\1\2', text)
    return text


def preprocess_sentence(sent_):
    sent_ = re.sub(r'\[.*\]', "", sent_)
    sent_ = join_split_words(sent_)
    sent_ = fix_break_lines(sent_)
    sent_ = clean_text(sent_)
    sent_ = separate_punctuation(sent_)
    sent_ = join_punctuation_marks(sent_)
    sent_ = replace_punct(sent_)
    sent_ = remove_before_sent(sent_)
    return sent_.lower()


def label2text(tokens_, labels_):
    """
    Retorna o texto de uma sentença a partir de seus tokens e labels
    :param tokens_: tokens da sentença
    :param labels_: labels da sentença
    :return:
    """
    text = ''
    for j, label in enumerate(labels_):

        if j == 0:
            text += tokens_[j]
        else:
            text += " " + tokens_[j]

        if label == 'I-PERIOD':
            text += f"."
        elif label == 'I-COMMA':
            text += ", "

    return text


def join_split_words(text):
    """
    Junta palavras separadas por um \n
    """

    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'(\w+)_\n(\w+)', r'\1\2', text)
    return text


def fix_break_lines(text):
    text = re.sub(r'/n', '\n', text)
    return text


def separate_punctuation(text):
    text = re.sub(r'([.,?!;:])(\w)', r'\1 \2', text)
    return text


def remove_before_sent(text):
    tokens = text.split()
    if len(tokens) > 0 and tokens[0] in [',', '.']:
        tokens.pop(0)

    return ' '.join(tokens)


def clean_text(text):
    """
    Remove caracteres especiais e espaços em branco e as
    marcações de início e fim de parágrafo e afins.
    :param text:
    :return:
    """
    text = re.sub(r'\[\?\}', '', text).strip()
    text = re.sub(special_pattern, ' ', text)
    text = re.sub(marks, '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r'\"', '', text).strip()
    text = re.sub(r'\[\?\}', '', text).strip()
    text = re.sub(r'[*+]', '', text)
    return ' '.join(text.split())


def replace_punct(text):
    text = re.sub(r'[!?;]', '.', text)
    text = re.sub(r'[:]', ',', text)
    return text

import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from nltk.tokenize import wordpunct_tokenize
import string
from utils.utils import text2labels
import os, jsonlines, re
from tqdm.notebook import tqdm
from datasets import load_dataset


def remove_punctuation(text):
    """
    Remove punctuation from text
    :param text: text to remove punctuation from
    :return:  text without punctuation
    """
    text = [word.lower() for word in wordpunct_tokenize(text)
            if word not in string.punctuation]
    return text


def preprocess_function(examples):
    labels = list(map(text2labels, examples['paraphrase']))
    words = list(map(remove_punctuation, examples["paraphrase"]))

    examples["tokens"] = words
    examples["labels"] = labels
    return examples


def join_punctuation_marks(text):
    ## join punctuation mark
    text = re.sub(r'(\w)\s([.,?!;:]+)', r'\1\2', text)
    return text


text = '; fantasminhas existem e ttêm medo de gente;'


def remove_initial_punct(text_):
    ## Remove punctuation from in front of the text

    i = 0
    tokens = wordpunct_tokenize(text_)

    while len(tokens) > 0 and tokens[i] in string.punctuation:
        tokens.pop(i)

    return ' '.join(tokens)


def replace_punctuation(text):
    new_text = re.sub(r'[;:!?]', '.', text)
    return new_text


def remove_extra_punctuation(text):
    ## Remove extra presence of punctuation

    text = re.sub(r'([.,?!;:])+', r'\1', text)
    return text


def remove_bad_symbols(text):
    new_text = ''.join([char for char in list(text) if char not in bad_chars])
    return new_text


def preprocess_pipeline(text):
    text = remove_initial_punct(text)
    text = join_punctuation_marks(text)
    text = remove_extra_punctuation(text)
    text = replace_punctuation(text)
    text = remove_bad_symbols(text)
    return text


DATASET_PATH = './dataset/'

os.makedirs(DATASET_PATH, exist_ok=True)
dataset_paths = ['/content/corpus_readability_nlp_portuguese/1_Ensino_Fundamental_I',
                 '/content/corpus_readability_nlp_portuguese/2_Ensino_Fundamental_II',
                 '/content/corpus_readability_nlp_portuguese/3_Ensino_Medio',
                 '/content/corpus_readability_nlp_portuguese/4_Ensino_Superior']
dataset_list = []
lines = []
for root_dir in dataset_paths:
    tag = os.path.split(root_dir)[-1]

    for filename in tqdm(os.listdir(root_dir)):
        with open(os.path.join(root_dir, filename), encoding='utf-8-sig') as f:
            text = f.read()
            new_text = preprocess_pipeline(text)
            for sentences in sent_tokenize(new_text):
                for sent in sent_tokenize(preprocess_pipeline(sentences).encode().decode('utf-8-sig')):
                    new_sent = ' '.join(sent.split())
                    real_tokens = [token for token in wordpunct_tokenize(new_sent)
                                   if token not in string.digits + string.punctuation]

                    if len(real_tokens) > 1 and new_sent not in lines:
                        lines.append(new_sent)
                        line = {
                            'text_id': int(filename.replace('_', '').replace('.txt', '')),
                            'text': new_sent,
                            'level': re.sub(r'\d_', '', tag)
                        }
                        dataset_list.append(line)
                        with jsonlines.open(os.path.join(DATASET_PATH, f'corpus_readability.jsonl'),
                                            mode='a') as writer:
                            writer.write(line)

dataset = load_dataset('json', data_dir='./dataset/')

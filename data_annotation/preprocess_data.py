from __future__ import annotations

import math
import string
from collections import defaultdict

import click
import datasets
import spacy
from spacy import tokens

from data_annotation.preprocess_text import preprocess_sentence, label2text
from data_annotation.util.quality import remove_bad_tokens
from util import dataset
from util.annotation import preprocess, comparison

nlp = spacy.load('pt_core_news_md')
nlp.add_pipe('sentencizer')

remover = preprocess.TokenRemover()
corrector = preprocess.TokenCorrector()
hyperseg = preprocess.HypersegmentationCorrector()
hyposeg = preprocess.HyposegmentationCorrector()


def set_sentences(doc: tokens.Doc) -> tokens.Doc:
    for tok in doc[:-1]:
        tok.is_sent_start = doc[tok.i - 1].text == '.'
    return doc


def scale_number(unscaled, to_min, to_max, from_min, from_max):
    return math.floor((to_max - to_min) * (unscaled - from_min) / (from_max - from_min) + to_min)


special_pattern = r'\s+|\n+|/n|\t+|-|—'
marks = r'\[\w{0,3}|\W{0,3}\]|\(|\)'


def get_ner_label(annot_, token_id):
    """
    Retorna o label de uma anotação de acordo com o token
    :param annot_: anotação
    :param token_id: id do token
    :return:
    """

    match_token = annot_.doc[token_id].text
    condition_1 = (match_token != "." and annot_.label_ == "Erro de Pontuação")
    condition_3 = (match_token != "," and annot_.label_ == "Erro de vírgula")

    condition_4 = (match_token == "," and annot_.label_ == "Erro de vírgula")
    condition_5 = (match_token == "." and annot_.label_ == "Erro de Pontuação")
    if condition_1:
        label = "I-PERIOD"
    elif condition_3:
        label = "I-COMMA"
    elif condition_4 or condition_5:
        label = "NO-PUNCT"
    else:
        raise ValueError(f"Condition for {match_token} and label {annot_.label_} not implemented ")

    return label


def load_raw_dataset(dataset_path: str, save_path: str = None):
    docs = dataset.create(dataset_path)
    print(*docs[0].spans.keys())
    print('Documentos:', len(docs))

    from tqdm import tqdm
    from spacy import tokens

    for i, doc in tqdm(enumerate(docs), total=len(docs)):
        doc = remove_bad_tokens(doc)
        doc = corrector(doc)
        doc = hyperseg(doc)
        docs[i] = hyposeg(doc)

    if save_path:
        # '../data/apa-nlp-span-clean-fix-hyper-hypo.docbin'
        tokens.DocBin(docs=docs, store_user_data=True).to_disk(save_path)
    return docs


@click.command()
@click.option('--dataset_path', type=click.Path(exists=True), default='data/usp-annotated.csv')
@click.option('--spacy_dataset_path', type=click.Path(exists=True),
              default='../data/apa-nlp-span-clean-fix-hyper-hypo.docbin')
@click.option('--save_path', type=click.Path(exists=False), default='../data/dataset-ner.json')
def main(dataset_path: str,
         spacy_dataset_path: str,
         path_to_save: str,
         push_to_hf: bool = False, ):
    from spacy import tokens

    if spacy_dataset_path:
        docs = list(tokens.DocBin().from_disk(spacy_dataset_path).get_docs(nlp.vocab))
    elif dataset_path:
        docs = load_raw_dataset(dataset_path)
    else:
        raise ValueError("You must provide a dataset path or a spacy dataset path")

    for doc in docs:
        doc = set_sentences(doc)

    results = []

    ner_dataset = []
    for text_id, doc in enumerate(docs):
        documento = defaultdict(set)
        spans = doc.spans['punctuation']

        ner_tags = ["O"] * len(doc)
        tokens = [token.text.lower() for token in doc]

        for annot in comparison.merge(list(doc.sents), spans):

            if len(preprocess_sentence(annot.sent.text).split()) < 3:
                continue
            if annot.label != 0:
                documento[annot.sent].add(annot)

        for sent, annots in documento.items():
            for annot in annots:
                ner_tags[annot.end - 1] = get_ner_label(annot, annot.end - 1)

        new_tokens = []
        new_ner_tags = []

        for i, (token, tag) in enumerate(zip(tokens, ner_tags)):
            if tag in ["I-PERIOD", "I-COMMA"]:

                if token not in [",", ".", ";", "!", "...", "?"]:
                    new_tokens.append(token)
                    new_ner_tags.append(tag)
                else:
                    if len(new_tokens) > len(new_ner_tags):
                        new_ner_tags.append(tag)
                    else:
                        try:
                            new_ner_tags[-1] = tag
                        except IndexError:
                            continue
            elif tag in ["NO-PUNCT"]:
                new_ner_tags.append("O")
                new_tokens.append(token)
            elif token in [","]:

                try:
                    new_ner_tags[-1] = "I-COMMA"
                except IndexError:
                    continue
            elif token in [".", ";", "!", "...", "?"]:
                try:
                    new_ner_tags[-1] = "I-PERIOD"
                except IndexError:
                    continue
            else:
                if token not in string.punctuation:
                    new_tokens.append(token)
                    new_ner_tags.append("O")

            if len(new_ner_tags) and new_tokens[0] in string.punctuation:
                new_tokens = new_tokens[1:]
                new_ner_tags = new_ner_tags[1:]

            if len(new_ner_tags) and new_ner_tags[-1] == "I-PERIOD":
                if len(new_tokens) != len(new_ner_tags):
                    print(doc.text)
                    print(list(zip(new_tokens, new_ner_tags)))

                ner_dataset.append({
                    'text_id': text_id,
                    'text': doc.text,
                    'labels': new_ner_tags,
                    'tokens': new_tokens,
                    'sent_text': label2text(new_tokens, new_ner_tags),
                    'tag': "both_annotators"
                })
                new_tokens = []
                new_ner_tags = []
        results.append(documento)

    import json

    json.dump(ner_dataset, open(path_to_save, 'w'), indent=4)

    if push_to_hf:
        datasets.load_dataset("json",
                              data_files={"train": path_to_save})


if __name__ == '__main__':
    main()

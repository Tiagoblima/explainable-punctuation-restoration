import os

import click
import pandas as pd
from datasets import load_dataset, Dataset
from seqeval.metrics import classification_report
from tqdm.notebook import tqdm
from transformers import pipeline

from seq2seq.evaluate_t5 import get_t5_pred_sentence
from seq_labeling.evalute_bert import get_bert_pred_sentence
from utils.utils import remove_punctuation
from utils.utils import text2labels

TEDTALK2012_BERT_BASE = 'tiagoblima/punctuation-tedtalk2012-bert-base'
TEDTALK2012_BERT_LARGE = 'tiagoblima/punctuation-tedtalk2012-bert-large'
TEDTALK2012_T5_BASE = 'tiagoblima/punctuation-tedtalk2012-t5-base'
TEDTALK2012_T5_LARGE = 'tiagoblima/punctuation-tedtalk2012-t5-large'
NILC_T5_BASE = 'tiagoblima/punctuation-nilc-t5-base'
NILC_T5_LARGE = 'tiagoblima/punctuation-nilc-t5-large'
NILC_BERT_BASE = 'tiagoblima/punctuation-bert-t5-base'
NILC_BERT_LARGE = 'tiagoblima/punctuation-bert-t5-large'
MEC_T5_BASE = 'tiagoblima/punctuation-mec-t5-base'
MEC_T5_LARGE = 'tiagoblima/punctuation-mec-t5-large'
MEC_BERT_BASE = 'tiagoblima/punctuation-mec-t5-base'
MEC_BERT_LARGE = 'tiagoblima/punctuation-mec-t5-large'

MODEL_NAMES = [TEDTALK2012_BERT_BASE, TEDTALK2012_BERT_LARGE, TEDTALK2012_T5_BASE, TEDTALK2012_T5_LARGE,
               NILC_T5_BASE, NILC_T5_LARGE, NILC_BERT_BASE, NILC_BERT_LARGE,
               MEC_T5_BASE, MEC_T5_LARGE, MEC_BERT_BASE, MEC_BERT_LARGE]


def get_model(model_path: str, model_type: str):
    if model_type == 'bert':
        model = pipeline("ner", model=model_path, aggregation_strategy="average", device=0)
    elif model_type == 't5':
        model = pipeline("text2text-generation", model_path, max_length=512, device=0, use_auth_token=True)
    else:
        raise ValueError("Model type not supported")

    return model


def compute_report(test_subset: Dataset,
                   model_path: str,
                   model_type: str = 'bert'):
    pred_labels = []
    true_labels = []

    model = get_model(model_path, model_type)
    for samples in tqdm(test_subset):

        text = ' '.join(remove_punctuation(' '.join(samples['tokens']))).lower()

        pred_pipeline = get_bert_pred_sentence
        if model_type == "t5":
            text = "Recognize Entities: " + text
            pred_pipeline = get_t5_pred_sentence

        pred_text = pred_pipeline(text, model)

        preds = text2labels(pred_text)
        true_label = [t_lbl for t_lbl, p_lbl in zip(samples['labels'], preds)]
        preds = [p_lbl for t_lbl, p_lbl in zip(samples['labels'], preds)]

        pred_labels.append(preds)
        true_labels.append(true_label)

    return classification_report(true_labels, pred_labels, output_dict=True)


@click.command()
@click.option('--model_name_or_path', type=str, default='tiagoblima/punctuation-tedtalk2012-bert-base')
@click.option('--dataset_name', type=str, default='tiagoblima/punctuation-tedtalk2012-bert')
@click.option('--eval_split', type=str, default='test')
@click.option('--model_type', type=str, default='bert')
@click.option('--output_path', type=str, default='.')
@click.option('--use_auth_token', type=bool, default=True)
def main(model_name_or_path: str,
         dataset_name: str,
         eval_split: str,
         model_type: str,
         output_path: str,
         use_auth_token: bool):
    eval_dataset = load_dataset(dataset_name,
                                split=eval_split,
                                use_auth_token=use_auth_token)
    report = compute_report(eval_dataset, model_name_or_path, model_type=model_type)
    df = pd.DataFrame.from_dict(report, orient='index')
    df.to_csv(os.path.join(output_path, f'{model_name_or_path.split("/")[-1]}_{eval_split}.csv'))

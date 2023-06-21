import jsonlines
import pandas as pd
from datasets import load_dataset

from seqeval.metrics import classification_report


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


eval_split = 'train'
eval_dataset = load_dataset('tiagoblima/punctuation-nilc-bert', split=eval_split)

pred_df = pd.read_csv('results/zero_shot/nilc/punctuation_predictions.csv')
pred_df['pred_labels'] = pred_df['pred_labels'].apply(eval)
pred_report = compute_scores(eval_dataset["labels"], pred_df['pred_labels'].tolist())
report_df = pd.DataFrame.from_dict(pred_report, orient='index')
report_df.to_csv('punctuation_report.csv')

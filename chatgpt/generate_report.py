import pandas as pd
from datasets import load_dataset

from chatgpt.utils import compute_scores

eval_split = 'train'
eval_dataset = load_dataset('tiagoblima/punctuation-nilc-bert', split=eval_split)

pred_df = pd.read_csv('results/chatgpt3-turbo/zero_shot/nilc/punctuation_predictions.csv')
pred_df['pred_labels'] = pred_df['pred_labels'].apply(eval)
pred_report = compute_scores(eval_dataset["labels"], pred_df['pred_labels'].tolist())
report_df = pd.DataFrame.from_dict(pred_report, orient='index')
report_df.to_csv('punctuation_report.csv')

import time

import jsonlines
import openai
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from chatgpt.constants import API_KEY
from chatgpt.utils import remove_punctuation, text2labels, compute_scores, chat_gpt_predict

eval_split = 'train'
mec_dataset = load_dataset('tiagoblima/punctuation-mec-bert-v2', split=eval_split)
nilc_dataset = load_dataset('tiagoblima/punctuation-nilc-bert', split=eval_split)
openai.api_key = API_KEY


def prepare_prompt(sent_text):
    new_text = " ".join(remove_punctuation(sent_text))

    prompt = f"corriga a seguinte frase colocando os sinais de 'ponto final' e 'vírgula' sem qualquer outra mudança:  '{new_text}'"
    prompt += "\n\nbons exemplos de anotação:\n"
    prompt += "\n".join(sample for sample in nilc_dataset[:20]["text"])

    return {"role": "user", "content": prompt}


predictions = []
text_column_name = "sent_text"
with jsonlines.open('results/20_shot/punctuation_predictions.jsonl', mode='w') as writer:
    for sent_text in tqdm(mec_dataset[text_column_name], total=len(mec_dataset[text_column_name])):
        messages = [prepare_prompt(sent_text)]

        pred_text = chat_gpt_predict(messages)
        writer.write({"pred_text": pred_text, "pred_labels": text2labels(pred_text)})
        predictions.append({"pred_text": pred_text, "pred_labels": text2labels(pred_text)})


pred_df = pd.DataFrame(predictions)

pred_df.to_csv('results/20_shot/punctuation_predictions.csv', index_label=False, index=False)
pred_report = compute_scores(mec_dataset["labels"], pred_df['pred_labels'].tolist())
report_df = pd.DataFrame.from_dict(pred_report, orient='index')
report_df.to_csv('results/20_shot/punctuation_report.csv')

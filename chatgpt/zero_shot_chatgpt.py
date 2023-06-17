import string
import time

import jsonlines
from datasets import load_dataset
import openai
import pandas as pd
from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm
from constants import API_KEY

from chatgpt.utils import remove_punctuation, text2labels

eval_split = 'train'
eval_dataset = load_dataset('tiagoblima/punctuation-mec-bert-v2', split=eval_split)




def prepare_prompt(sent_text):
    return {"role": "user", "content": " ".join(remove_punctuation(sent_text))}


StartPrompt = "coloque os sinais de 'ponto final' e 'vírgula' na seguinte sentença sem qualquer outra correção:"
i = 1
predictions = []

with jsonlines.open('results/zero_shot/punctuation_predictions.jsonl', mode='w') as writer:
    for sent_text in tqdm(eval_dataset["sent_text"], total=len(eval_dataset["sent_text"])):
        messages = [{"role": "system", "content": StartPrompt}, prepare_prompt(sent_text)]

        while True:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
            except openai.error.RateLimitError:
                time.sleep(20)
                continue
            break

        i += 1

        writer.write({"pred_text": pred_text, "pred_labels": text2labels(pred_text)})
        predictions.append({"pred_text": pred_text, "pred_labels": text2labels(pred_text)})
        time.sleep(20)

pred_df = pd.DataFrame(predictions)

pred_df.to_csv('punctuation_predictions.csv', index_label=False, index=False)


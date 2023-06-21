import os

import jsonlines
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from chatgpt.utils import text2labels, chat_gpt_predict


def main(dataset_name: str,
         save_path: str,
         api_key: str,
         split_name: str = 'train',
         text_column_name: str = 'text'):
    prompt = "coloque os sinais de 'ponto final' e 'vírgula' na seguinte sentença sem qualquer outra correção:"

    predictions = []

    dataset = load_dataset(dataset_name, split=split_name)

    path_to_file = f'{save_path}/punctuation_predictions.jsonl'
    continue_from_checkpoint = os.path.isfile(path_to_file)

    with jsonlines.open(path_to_file, mode='a') as writer:
        if continue_from_checkpoint:
            with jsonlines.open(path_to_file) as reader:
                for i, line in enumerate(reader):
                    predictions.append(line)

        for sent_text in tqdm(dataset[text_column_name][i:], total=len(dataset[text_column_name]) - i):
            prompt += f"\n\n'{sent_text}'"
            pred_text = chat_gpt_predict(prompt, api_key)
            writer.write({"pred_text": pred_text, "pred_labels": text2labels(pred_text)})
            predictions.append({"pred_text": pred_text, "pred_labels": text2labels(pred_text)})

    pred_df = pd.DataFrame(predictions)

    pred_df.to_csv('punctuation_predictions.csv', index_label=False, index=False)

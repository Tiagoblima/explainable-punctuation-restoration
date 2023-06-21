import os

import click
import jsonlines
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from chatgpt.utils import text2labels, chat_gpt_predict, compute_scores

PROMPT = "coloque os sinais de 'ponto final' e 'vírgula' na seguinte sentença sem qualquer outra correção:"

load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY_GPT3.5TURBO')


@click.command()
@click.option('--dataset_name', default='tiagoblima/punctuation-mec-bert-v2', help='Dataset name')
@click.option('--prompt', default=PROMPT, help='Prompt to be used')
@click.option('--save_path', default='results/chatgpt3-turbo/20_shot', help='Path to save results')
@click.option("--api_key", default=API_KEY, help="OpenAI API key")
@click.option('--split_name', default='train', help='Dataset split name')
@click.option('--text_column_name', default='sent_text', help='Dataset text column name')
def main(dataset_name: str,
         prompt: str,
         save_path: str,
         api_key: str,
         split_name: str = 'train',
         text_column_name: str = 'text'):
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
            sent_prompt = prompt + f"\n\n'{sent_text}'"
            pred_text = chat_gpt_predict(prompt, api_key)
            writer.write({"pred_text": pred_text, "pred_labels": text2labels(pred_text)})
            predictions.append({"pred_text": pred_text, "pred_labels": text2labels(pred_text)})

    pred_df = pd.DataFrame(predictions)
    pred_report = compute_scores(dataset["labels"], pred_df['pred_labels'].tolist())
    report_df = pd.DataFrame.from_dict(pred_report, orient='index')
    report_df.to_csv(os.path.join(save_path, 'punctuation_report.csv'))
    pred_df.to_csv(os.path.join(save_path, 'punctuation_predictions.csv'), index_label=False, index=False)


if __name__ == '__main__':
    main()

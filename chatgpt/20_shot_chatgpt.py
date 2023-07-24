import os.path
import click
import jsonlines
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from chatgpt.utils import remove_punctuation, text2labels, compute_scores, chat_gpt_predict

load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY_GPT3.5TURBO')

PROMPT = f"corrija a seguinte frase colocando os sinais de 'ponto final' e 'vírgula' " \
         f"sem qualquer outra mudança:  "

NEW_PROMPT = "Act like punctuation corrector in brazilian portuguese: Place the 'period' and 'comma' punctuation marks " \
             "in the following sentence without any other corrections:"

def prepare_prompt(text, examples):
    prompt = PROMPT + f"'{remove_punctuation(text)}'"
    prompt += "\n\nHere are some good annotation examples:\n"
    prompt += "\n".join(sample for sample in examples)

    return prompt


@click.command()
@click.option('--train_dataset_name', default='tiagoblima/punctuation-nilc-bert', help='Dataset name')
@click.option('--eval_dataset_name', default='tiagoblima/punctuation-mec-bert-v2', help='Dataset name')
@click.option('--save_path', default='results/chatgpt3-turbo/20_shot', help='Path to save results')
@click.option("--api_key", default=API_KEY, help="OpenAI API key")
@click.option('--num_samples', default=5, help='Number of samples to pass ')
@click.option('--train_split_name', default='train', help='Dataset split name')
@click.option('--eval_split_name', default='train', help='Dataset split name')
@click.option('--text_column_name', default='text', help='Dataset text column name')
def main(train_dataset_name: str,
         eval_dataset_name: str,
         save_path: str,
         api_key: str,
         num_samples: int,
         train_split_name: str = 'train',
         eval_split_name: str = 'train',
         text_column_name: str = 'text'):
    predictions = []
    path_to_file = f'{save_path}/punctuation_predictions.jsonl'
    continue_from_checkpoint = os.path.isfile(path_to_file)

    train_dataset = load_dataset(train_dataset_name, split=train_split_name)
    eval_dataset = load_dataset(eval_dataset_name, split=eval_split_name)

    with jsonlines.open(path_to_file, mode='a') as writer:
        i = 0
        if continue_from_checkpoint:
            with jsonlines.open(path_to_file) as reader:
                for i, line in enumerate(reader):
                    predictions.append(line)

        for sent_text in tqdm(eval_dataset["sent_" + text_column_name][i:11],
                              total=len(eval_dataset[text_column_name][:11]) - i):

            prompt = prepare_prompt(sent_text, train_dataset[text_column_name][:num_samples])
            pred_text = chat_gpt_predict(prompt, api_key)
            writer.write({"pred_text": pred_text, "pred_labels": text2labels(pred_text)})
            predictions.append({"pred_text": pred_text, "pred_labels": text2labels(pred_text)})

    pred_df = pd.DataFrame(predictions)
    pred_report = compute_scores(eval_dataset["labels"], pred_df['pred_labels'].tolist())
    report_df = pd.DataFrame.from_dict(pred_report, orient='index')
    report_df.to_csv(os.path.join(save_path, 'punctuation_report.csv'))
    pred_df.to_csv(os.path.join(save_path, 'punctuation_predictions.csv'), index_label=False, index=False)


if __name__ == '__main__':
    main()

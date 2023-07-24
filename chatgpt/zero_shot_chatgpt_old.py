import os
from itertools import chain

import click
import jsonlines
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from chatgpt.utils import text2labels, chat_gpt_predict, compute_scores, remove_punctuation

PROMPT = "coloque os sinais de 'ponto final' e 'vírgula' na seguinte sentença sem qualquer outra correção:"
NEW_PROMPT = "Act like punctuation corrector in brazilian portuguese: Place the 'period' and 'comma' punctuation marks " \
             "in the following sentence without any other corrections:"
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')


@click.command()
@click.option('--dataset_name', default='tiagoblima/punctuation-mec-bert-v2', help='Dataset name')
@click.option('--prompt', default=NEW_PROMPT, help='Prompt to be used')
@click.option('--model', default='gpt-3.5-turbo', help='Prompt to be used')
@click.option('--save_path', default='results/gpt-3.5-turbo/zero_shot/', help='Path to save results')
@click.option("--api_key", default=API_KEY, help="OpenAI API key")
@click.option('--split_name', default='train', help='Dataset split name')
@click.option('--text_column_name', default='sent_text', help='Dataset text column name')
def main(dataset_name: str,
         prompt: str,
         model: str,
         save_path: str,
         api_key: str,
         split_name: str = 'train',
         text_column_name: str = 'text'):
    predictions = []

    dataset = load_dataset(dataset_name, split=split_name)
    os.makedirs(save_path, exist_ok=True)
    path_to_file = f'{save_path}/punctuation_predictions.jsonl'
    continue_from_checkpoint = os.path.isfile(path_to_file)
    open(os.path.join(save_path, "prompt.txt"), "w").write(prompt)
    with jsonlines.open(path_to_file, mode='a') as writer:
        i = 0
        if continue_from_checkpoint:
            with jsonlines.open(path_to_file) as reader:
                for i, line in enumerate(reader):
                    predictions.append(line)

        def calculate_cost():
            list_of_words = list(
                map(lambda x: (prompt + f"'{remove_punctuation(x)}'").split(), dataset[text_column_name]))
            total_words_prompt = len(list(chain.from_iterable(list_of_words)))
            total_input_tokens = ((total_words_prompt * 1000) / 750)
            total_input_cost = (total_input_tokens/1000) * 0.03

            list_of_words = list(map(lambda x: f"'{remove_punctuation(x)}'".split(), dataset[text_column_name]))
            output = len(list(chain.from_iterable(list_of_words)))
            total_output_tokens = ((output * 1000) / 750)
            total_output_cost = (total_output_tokens/1000) * 0.06

            return total_input_cost, total_output_cost

        total_input_cost, total_output_cost = calculate_cost()

        print(f"Total input cost, words + prompt: ${total_input_cost}")
        print(f"Total output cost: ${total_output_cost}")
        print(f"Total cost: ${total_output_cost+total_input_cost}")

        for sent_text in tqdm(dataset[text_column_name][i:], total=len(dataset[text_column_name]) - i):
            sent_prompt = prompt + f"'{remove_punctuation(sent_text)}'"

            pred_text = chat_gpt_predict(sent_prompt, api_key, model=model)

            writer.write({"sent_text": sent_text, "pred_text": pred_text, "pred_labels": text2labels(pred_text)})
            predictions.append({"pred_text": pred_text, "pred_labels": text2labels(pred_text)})

    pred_df = pd.DataFrame(predictions)
    pred_report = compute_scores(dataset["labels"], pred_df['pred_labels'].tolist())
    report_df = pd.DataFrame.from_dict(pred_report, orient='index')
    report_df.to_csv(os.path.join(save_path, 'punctuation_report.csv'))
    pred_df.to_csv(os.path.join(save_path, 'punctuation_predictions.csv'), index_label=False, index=False)


if __name__ == '__main__':
    main()

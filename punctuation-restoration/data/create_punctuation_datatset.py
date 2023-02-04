import os
import re

import click
from datasets import load_dataset
from utils.utils import text2labels, text2t5labels

import jsonlines


def build_dataset(dataset, save_path, data_format):
    for split in dataset.column_names:

        data = dataset[split]['text']

        i = 0

        with jsonlines.open(os.path.join(save_path, split + '.jsonl'), 'w') as f:
            for line in data:
                text = re.sub(r'[!;]', '.', line)
                text = re.sub(r'[:]', ',', text)
                text = re.sub(r'\s[-]\s', ',', text).lower()

                emotions = re.findall(r'\(\w+\)', text)
                if len(emotions) > 0:
                    continue

                item = {"text_id": i}

                if data_format == "bert":
                    item.update(text2labels(text))
                elif data_format == "t5":
                    item.update(text2t5labels(text))
                else:
                    raise ValueError("Unknown data format")
                i += 1
                f.write(item)


@click.command()
@click.option('--path_to_save', type=str, default='./punctuation-restoration/data')
@click.option('--dataset_name', type=str, default='tiagoblima/tedtalk2012-03')
@click.option('--save_dataset', type=str, default='tiagoblima/punctuation-tedtalk2012')
@click.option('--data_format', type=str, default="t5", help="bert or t5", )
@click.option('--private', type=bool, default=True)
@click.option('--use_auth_token', type=bool, default=True)
def main(path_to_save, dataset_name, save_dataset, data_format, private=True, use_auth_token=True):
    os.makedirs(path_to_save, exist_ok=True)

    dataset = load_dataset(dataset_name, use_auth_token=use_auth_token)

    build_dataset(dataset, path_to_save, data_format)

    new_dataset = load_dataset('json', data_files={'train': os.path.join(path_to_save, 'train.jsonl'),
                                                   'validation': os.path.join(path_to_save, 'validation.jsonl'),
                                                   'test': os.path.join(path_to_save, 'test.jsonl')})
    if save_dataset.split('-')[-1] != data_format:
        save_dataset = save_dataset + '-' + data_format

    new_dataset.push_to_hub(save_dataset, private=private)


if __name__ == '__main__':
    main()

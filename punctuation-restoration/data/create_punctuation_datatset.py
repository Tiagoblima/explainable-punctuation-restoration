import os
import string

import click
import tqdm
from datasets import load_dataset
from utils.utils import text2labels, text2t5labels
from nltk.tokenize import sent_tokenize
from utils.preprocess import preprocess_text

import jsonlines


def build_dataset(dataset, save_path, data_format, split_in_sentence):
    for split in dataset.column_names:

        dataset_split = dataset[split]

        i = 0

        with jsonlines.open(os.path.join(save_path, split + '.jsonl'), 'w') as f:
            for indx in tqdm.tqdm(range(dataset_split.num_rows)):

                item = dataset_split[indx]
                raw_text = item['text']

                text = preprocess_text(raw_text)
                if split_in_sentence:
                    try:
                        for sent_text in sent_tokenize(text):

                            if sent_text in string.punctuation:
                                print(f"Skip punctuation {sent_text}")
                                continue
                            item.update({
                                "sent_text": sent_text,
                            })
                            if data_format == "bert":

                                item.update(text2labels(sent_text))
                            elif data_format == "t5":
                                item.update(text2t5labels(sent_text))
                            else:
                                raise ValueError("Unknown data format")
                            f.write(item)
                    except ValueError:
                        print(f"Can't tokenize sentence {text}")
                        breakpoint()
                    i += 1
                else:
                    item.update({
                        "text": text,
                    })
                    if data_format == "bert":

                        item.update(text2labels(text))
                    elif data_format == "t5":
                        item.update(text2t5labels(text))
                    else:
                        raise ValueError("Unknown data format")
                    f.write(item)


@click.command()
@click.option('--path_to_save', type=str, default='./punctuation-restoration/data')
@click.option('--dataset_name', type=str, default='tiagoblima/tedtalk2012-03')
@click.option('--save_dataset', type=str, default='tiagoblima/punctuation-tedtalk2012-full-text')
@click.option('--data_format', type=str, default="bert", help="bert or t5", )
@click.option('--split_in_sentences', is_flag=True, type=bool, default=False, help="Wether to split in sentences or not", )
@click.option('--private', type=bool, default=True)
@click.option('--use_auth_token', type=bool, default=True)
def main(path_to_save,
         dataset_name,
         save_dataset: str,
         data_format: str,
         split_in_sentences: bool,
         private: bool = True,
         use_auth_token: bool = True):
    os.makedirs(path_to_save, exist_ok=True)

    dataset = load_dataset(dataset_name, use_auth_token=use_auth_token)

    build_dataset(dataset, path_to_save, data_format, split_in_sentences)

    new_dataset = load_dataset('json', data_files={'train': os.path.join(path_to_save, 'train.jsonl'),
                                                   'validation': os.path.join(path_to_save, 'validation.jsonl'),
                                                   'test': os.path.join(path_to_save, 'test.jsonl')})
    if save_dataset.split('-')[-1] != data_format:
        save_dataset = save_dataset + '-' + data_format

    new_dataset.push_to_hub(save_dataset, private=private)


if __name__ == '__main__':
    main()

# importing the "tarfile" module
import os
import tarfile

import click
import jsonlines
import tqdm
from bs4 import BeautifulSoup as bs
from datasets import load_dataset
from utils.preprocess import preprocess_text


def build_dataset(output_root, folder_path, dict_tedtalk):
    i = 0
    folder_path = os.path.join(folder_path, 'pt-br-en', 'pt-br-en')
    for file in os.listdir(folder_path):

        if file.endswith(".pt-br") or file.endswith('.pt-br.xml'):

            with open(os.path.join(folder_path, file), errors='ignore') as f:
                bs_content = bs(f.read(), "lxml")

                transcripts = bs_content.find_all('transcript')

                dts = file.split('.')[-4]
                split = dict_tedtalk[dts]

                if not transcripts:
                    segs = bs_content.find_all('seg')
                    transcripts = segs
                with jsonlines.open(os.path.join(output_root, split + '.jsonl'), mode='w') as writer:
                    for transp in tqdm.tqdm(transcripts, total=len(transcripts)):
                        writer.write({
                            'text': preprocess_text(transp.get_text().strip()),
                            'lang': 'pt',
                            'id': i,
                        })
                        i += 1


@click.command()
@click.option('--output_root', default='tedtalk2012-03/', help='Output root path')
@click.option('--folder_path', default='../archives/pt-br-en/pt-br-en', help='Folder path')
@click.option('--dict_tedtalk', default='{"dev2010": "validation", "train": "train", "tst2010": "test"}',
              help='Dict tedtalk')
@click.option('--input_path', default='../archives/pt-br-en.tgz', help='Input path')
def main(output_root, folder_path, dict_tedtalk, input_path):
    try:
        os.makedirs(output_root)
    except OSError:
        pass

    # open file
    file = tarfile.open(input_path)

    output_path = input_path.split('.tgz')[0]
    # extracting file

    file.extractall(output_path)
    # close file
    file.close()

    build_dataset(output_root, folder_path, eval(dict_tedtalk))

    dataset = load_dataset('json', data_files={'train': os.path.join(output_root, 'train.jsonl'),
                                               'validation': os.path.join(output_root, 'validation.jsonl'),
                                               'test': os.path.join(output_root, 'test.jsonl')})
    dataset.push_to_hub('tiagoblima/tedtalk2012-03', private=True)
    print(dataset)


if __name__ == '__main__':
    main()

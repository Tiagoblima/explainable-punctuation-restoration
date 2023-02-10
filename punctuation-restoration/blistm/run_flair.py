import os
import shutil
import zipfile
from urllib import request
from urllib.error import HTTPError

import progressbar
import wandb
from flair.data import Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.optim import SGDW
import pandas as pd
from gensim.models import KeyedVectors

from evaluate import evaluate
from preprocess import preprocess
from utils import generate_test_file
import argparse

WANDB_KEY = '8e593ae9d0788bae2e0a84d07de0e76f5cf3dcf4'

embeddings = {
    'skip_s300': "http://143.107.183.175:22980/download.php?file=embeddings/word2vec/skip_s300.zip",
    'glove': "http://143.107.183.175:22980/download.php?file=embeddings/glove/glove_s300.zip"
}


class MyProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def run_train(trainer, args):
    wandb.login(key=args.wandb_key)
    with wandb.init(project=args.wandb_project) as run:
        run.name = f'bilstm_{args.embeddings}-{args.dataset}'
        trainer.train(args.model_dir, optimizer=SGDW, learning_rate=0.1, mini_batch_size=args.batch_size,
                      max_epochs=args.n_epochs)


def k_folding(args):
    print('\nRunning k-fold evaluation...')
    results_ents, results_micro_avg = [], []
    for folder in sorted(os.listdir(args.base_dir)):

        if os.path.isdir(os.path.join(args.base_dir, folder)):
            print(f'\nRunning on {folder}')
            dataset_path = os.path.join(args.base_dir, folder)
            out_path = os.path.join(args.path_to_data, folder)

            print('\nCleaning up previous runs...')
            shutil.rmtree(args.model_dir, ignore_errors=True)
            os.makedirs(out_path, exist_ok=True)

            print(f'\nPreprocessing {dataset_path}')
            preprocess(dataset_path, out_path)  # preprocess dataset

            corpus = ColumnCorpus(out_path, args.columns)
            # # filter empty sentences
            # corpus.filter_empty_sentences()
            # Create a new run
            project = "punctuation-blstm" + args.dataset

            tag_type = 'ner'

            tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)
            tag_dictionary.remove_item('<unk>')
            print('\nTags: ', tag_dictionary.idx2item)

            test_results_file = os.path.join(args.model_dir, 'test.tsv')
            new_test_file = os.path.join(args.model_dir, '_conlleval_test.tsv')
            generate_test_file(test_results_file, new_test_file)
            micro_avg, per_ents = evaluate(corpus, os.path.join(args.model_dir, 'best-model.pt'))
            micro_avg.update({'dataset_name': folder, 'classifier_name': 'bi-lstm'})
            micro_avg.pop('support')

            per_ents.update({'dataset_name': folder, 'classifier_name': 'bi-lstm'})

            results_micro_avg.append(micro_avg)
            results_ents.append(pd.DataFrame(per_ents))

    os.makedirs('./outputs/', exist_ok=True)
    pd.DataFrame(results_micro_avg).to_csv('./outputs/micro_avg.csv')
    pd.concat(results_ents).to_csv('./outputs/micro_avg_ents.csv')


def download_embeddings(args):
    try:

        print("Downloading embeddings...")
        url = embeddings.get(args.embedding_name, None)

        if url is None:
            print(f'Embedding {args.embedding_name} not found')
            return

        response = request.urlretrieve(url, args.embedding_name + ".zip", MyProgressBar())
    except HTTPError as e:
        print(f'In func my_open got HTTP {e.code} {e.reason}')

    print("Unzipping embeddings...")

    with zipfile.ZipFile(args.embedding_name + ".zip", 'r') as zip_ref:
        zip_ref.extractall(args.embedding_path)


def convert_embeddings(args):
    embeddings = KeyedVectors.load_word2vec_format(os.path.join(args.embedding_path, *os.listdir(args.embedding_path)), binary=False)

    embeddings.save(args.embeddings_bin_file)


def train(args):
    corpus_name = args.dataset

    embedding_name = args.embeddings
    embedding_types = []

    print(f'\nRunning using {args.embeddings}')

    if embedding_name == 'skip_s300':

        if not os.path.exists(args.embeddings_bin_file):
            download_embeddings(args)
            convert_embeddings(args)

        traditional_embedding = WordEmbeddings(args.embeddings_bin_file)

        if traditional_embedding is not None:
            embedding_types.append(traditional_embedding)

        embedding_name = args.embeddings.split('/')[-1].split('.')[0]
        args.model_dir += f'_{embedding_name}'

    elif embedding_name == 'bert':
        bert_embedding = TransformerWordEmbeddings('neuralmind/bert-base-portuguese-cased', layers='-1',
                                                   layer_mean=False)
        embedding_types.append(bert_embedding)
        args.model_dir += f'_{embedding_name}'
        sentence = Sentence('The grass is green.')
        bert_embedding.embed(sentence)
        print(f'Embedding size: {sentence[0].embedding.size()}')

    embeddings = StackedEmbeddings(embeddings=embedding_types)
    if args.use_crf:
        args.model_dir += '_crf'
        print('\nRunning using CRF')

    model_dir = os.path.join(args.model_dir, corpus_name)

    os.makedirs(model_dir, exist_ok=True)
    columns = {0: 'token', 1: args.tag_type}

    corpus = ColumnCorpus(args.path_to_data, columns)

    print('Train: ', corpus.train[0].to_tagged_string('label'))

    print('Dev: ', corpus.dev[0].to_tagged_string('label'))

    print('Test: ', corpus.test[0].to_tagged_string('label'))

    tag_type = 'ner'

    tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)

    tag_dictionary.remove_item('<unk>')

    print('\nTags: ', tag_dictionary.idx2item)

    tagger = SequenceTagger(hidden_size=args.hidden_size, embeddings=embeddings, tag_dictionary=tag_dictionary,
                            tag_type=args.tag_type, use_crf=args.use_crf)

    trainer = ModelTrainer(tagger, corpus)

    wandb.login(key=args.wandb_key)

    run_train(trainer, args)

    wandb.save(model_dir + '/*')
    test_results_file = os.path.join(model_dir, 'test.tsv')
    new_test_file = os.path.join(model_dir, corpus_name + '_conlleval_test.tsv')
    generate_test_file(test_results_file, new_test_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process dataframe data.')

    parser.add_argument('--result_path',
                        default='./results/',
                        help='output filename')

    parser.add_argument('--path_to_data',
                        default='./data/tedtalk2012',
                        help='Files must be a dataframe with headers sentence_id,words,label')

    parser.add_argument('--hidden_size',
                        default=512,
                        help='Files must be a dataframe with headers sentence_id,words,label')

    parser.add_argument('--dataset',
                        default='tedtalk2012',
                        help='Files must be a dataframe with headers sentence_id,words,label')

    parser.add_argument('--embeddings', default='skip_s300',
                        help='It must one of such models valid bert model, see hugginface plataform.')

    parser.add_argument('--batch_size', default=4,
                        help='Batch size for training.')

    parser.add_argument('--k_fold_eval',
                        action='store_true',
                        default=False,
                        help='Files must be a dataframe with headers sentence_id,words,label')

    parser.add_argument('--embeddings_bin_file', default='./embeddings', help='Embedding path')

    parser.add_argument('--use_crf', default=True, action='store_true')

    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')

    parser.add_argument('--tag_type', default='ner', help='Tag type')

    parser.add_argument('--model_dir', default='./models/bilstm', help='Model directory')

    parser.add_argument('--wandb', default=True, action='store_true', help='Wandb')

    parser.add_argument('--wandb_project', default='punctuation-restoration-kfold', help='Wandb project name')

    parser.add_argument('--wandb_key', default=WANDB_KEY, help='Wandb Key')

    parser.add_argument('--wandb_run_name', default='bilstm', help='Wandb run name')

    parser.add_argument('--wandb_tags', default=['bilstm', 'ner'], help='Wandb tags')

    args = parser.parse_args()
    train(args)

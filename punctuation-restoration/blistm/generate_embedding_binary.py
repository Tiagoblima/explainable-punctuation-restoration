from gensim.models import KeyedVectors
import argparse

parser = argparse.ArgumentParser(description='Process dataframe data.')
if __name__ == '__main__':
    parser.add_argument('--embeddings_txt_file', default='./embeddings/skip_s300.txt')
    parser.add_argument('--embeddings_bin_file', default='./embeddings/skip_s300.gensim')
    args = parser.parse_args()

    print('\nConverting TXT Embedding to Binary ...')
    embeddings = KeyedVectors.load_word2vec_format(args.embeddings_txt_file, binary=False)

    embeddings.save(args.embeddings_bin_file)

    print('\nProcess Completed ...')

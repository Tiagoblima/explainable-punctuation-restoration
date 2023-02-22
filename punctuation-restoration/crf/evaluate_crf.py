import argparse
import pickle

from seqeval.metrics import classification_report

from utils import read_corpus_file, data_preprocessing, convert_data, dump_report

parser = argparse.ArgumentParser(description='Process dataframe data.')

def main(model_file, path_to_data, report_dir, report_file, args):
    test_data = read_corpus_file(test_file, split_char=',')
    # Load dataset
    test_data = data_preprocessing(test_data)

    X_train, y_train = convert_data(train_data)
    X_test, y_test = convert_data(test_data)

    crf = pickle.load(open(model_file, 'rb'))
    y_pred = crf.predict(X_test)

    dict_report = classification_report(y_test, y_pred, output_dict=True)

    data_conll = ''

    for data, real_tags, pred_tags in zip(test_data, y_test, y_pred):
        words = data[0]
        sent = '\n'.join('{0} {1} {2}'.format(word, real_tag, pred_tag)
                         for word, real_tag, pred_tag in zip(words, real_tags, pred_tags))
        sent += '\n\n'
        data_conll += sent

    print('\nReport:', dict_report)

    print('\nSaving the report in:', report_file)

    dump_report(dict_report, report_file)

    script_result_file = os.path.join(report_dir, args.corpus_name + '_crf.tsv')

    with open(script_result_file, 'w', encoding='utf-8') as file:
        file.write(data_conll)


if __name__ == '__main__':
    parser.add_argument('--result_path',
                        default='./results/',
                        help='output filename')

    parser.add_argument('--path_to_data',
                        default='./data/tedtalk2012',
                        help='Files must be a dataframe with headers sentence_id,words,label')

    parser.add_argument('--k_fold_eval',
                        action='store_true',
                        default=False,
                        help='Files must be a dataframe with headers sentence_id,words,label')

    args = parser.parse_args()
    run(args)

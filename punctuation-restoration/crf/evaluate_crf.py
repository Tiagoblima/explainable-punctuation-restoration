import argparse
import os
import pickle

from seqeval.metrics import classification_report

from utils import read_corpus_file, data_preprocessing, convert_data, dump_report

parser = argparse.ArgumentParser(description='Process dataframe data.')


def evaluate(x_test, y_test, model_path):
    crf = pickle.load(open(model_path, 'rb'))
    y_pred = crf.predict(x_test)
    dict_report = classification_report(y_test, y_pred, output_dict=True)
    return dict_report, y_pred


def build_report(test_data, y_test, y_pred):
    data_conll = ''

    for data, real_tags, pred_tags in zip(test_data, y_test, y_pred):
        words = data[0]
        sent = '\n'.join('{0} {1} {2}'.format(word, real_tag, pred_tag)
                         for word, real_tag, pred_tag in zip(words, real_tags, pred_tags))
        sent += '\n\n'
        data_conll += sent
    return data_conll


def main(args):
    test_file = os.path.join(args.path_to_data, args.filename)
    test_data = read_corpus_file(test_file, split_char=',')
    # Load dataset
    test_data = data_preprocessing(test_data)
    X_test, y_test = convert_data(test_data)

    dict_report, y_pred = evaluate(X_test, y_test, args.model_path)

    os.makedirs(args.report_path, exist_ok=True)
    args.report_path = os.path.join(args.report_path, os.path.split(args.path_to_data)[-1])

    print('\nReport:', dict_report)

    print('\nSaving the report in:', args.report_path)

    dump_report(dict_report, os.path.join(args.report_path, 'report.csv'))

    script_result_file = os.path.join(args.report_path, 'data_conll.tsv')

    data_conll = build_report(test_data, y_test, y_pred)
    with open(script_result_file, 'w', encoding='utf-8') as file:
        file.write(data_conll)


if __name__ == '__main__':
    parser.add_argument('--model_path',
                        default='./results/',
                        help='output filename')

    parser.add_argument('--path_to_data',
                        default='./data/tedtalk2012',
                        help='Files must be a dataframe with headers sentence_id,words,label')

    parser.add_argument('--report_path',
                        default='./results/',
                        help='Path to final report')

    parser.add_argument('--filename',
                        default='test.csv',
                        help='Filename of the dataset')
    args = parser.parse_args()
    main(args)

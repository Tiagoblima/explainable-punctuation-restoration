import json
import os

import click
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger


def evaluate(corpus, tagger: SequenceTagger):
    result = tagger.evaluate(corpus.test, gold_label_type='ner')
    print(result.detailed_results)
    clf_report = result.classification_report

    clf_report.pop('weighted avg')
    clf_report.pop('macro avg')

    return clf_report['micro avg'], clf_report


@click.command()
@click.option('--path_to_model', default='./results/bilstm-model.pt', help='Path to model')
@click.option('--path_to_data', default='./data/tedtalk2012', help='Path to data')
@click.option('--report_path', default='./results/', help='Path to final report')
def main(
        path_to_model: str,
        path_to_data: str,
        report_path: str
):
    columns = {0: 'text', 1: 'ner'}
    corpus: ColumnCorpus = ColumnCorpus(path_to_data, columns,
                                        train_file='train.txt',
                                        test_file='test.txt',
                                        dev_file='dev.txt')
    model = SequenceTagger.load(path_to_model)
    clf_report, report = evaluate(corpus, model)
    os.makedirs(report_path, exist_ok=True)
    json.dump(report, open(os.path.join(report_path, 'report.json'), 'w'))
    json.dump(clf_report, open(os.path.join(report_path, 'clf_report.json'), 'w'))
    print(clf_report)
    print(report)


if __name__ == '__main__':
    main()

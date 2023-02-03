from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger


def evaluate(corpus, path_to_model='./tedtalk2012/best-model.pt'):

    tagger: SequenceTagger = SequenceTagger.load(path_to_model)
    result = tagger.evaluate(corpus.test, gold_label_type='ner')
    print(result.detailed_results)
    clf_report = result.classification_report

    clf_report.pop('weighted avg')
    clf_report.pop('macro avg')

    return clf_report['micro avg'], clf_report

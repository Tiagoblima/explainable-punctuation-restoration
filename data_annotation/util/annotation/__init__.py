from __future__ import annotations
from . import comparison, category_filters
from spacy import tokens
from typing import Callable

def merge(docs: list[tokens.Doc], output: str, scorer: Callable[[tokens.Span], float] = len) -> None:
    """Merge annotations on doc.spans[output] using """
    for doc in docs:
        annotations = comparison.merge(doc.spans['annot1'], doc.spans['annot2'], scorer=scorer)
        doc.spans.pop('annot1')
        doc.spans.pop('annot2')
        doc.spans[output] = annotations


def categorize(docs: list[tokens.Doc], span_key: str) -> None:
    """Create categories of annotations from doc.spans[span_key]"""
    for doc in docs:
        doc.spans['narrative'] = category_filters.narrative(doc.spans[span_key])
        doc.spans['punctuation'] = category_filters.punctuation(doc.spans[span_key])
        doc.spans['orthography'] = category_filters.orthography(doc.spans[span_key])
        doc.spans['plagiarism'] = category_filters.plagiarism(doc.spans[span_key])
        doc.spans['storyline'] = category_filters.storyline(doc.spans[span_key])
        doc.spans['semantic'] = category_filters.semantic(doc.spans[span_key])
        doc.spans['syntax'] = category_filters.syntax(doc.spans[span_key])
        doc.spans['stylistics'] = category_filters.stylistics(doc.spans[span_key])
        doc.spans['morphology'] = category_filters.morphology(doc.spans[span_key])
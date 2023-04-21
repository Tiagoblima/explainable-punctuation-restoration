from __future__ import annotations
from itertools import zip_longest
from typing import Callable
from spacy import tokens
from .annotation import category_filters


def log(text: str):
    print(text.center(80, '-'))


def span_info(span: tokens.Span | None) -> str:
    if span is None: return ''
    return f'[{span.start:02d}:{span.end:02d}, {span.label_:.14s}] {span.text!r}'


def span_pairs(annot1: list[tokens.Span], annot2: list[tokens.Span]) -> None:
    for s1, s2 in zip_longest(annot1, annot2):
        print('{:50.50s} | {:50.50s}'.format(span_info(s1), span_info(s2)))


def annotations(doc: tokens.Doc, filter_func: Callable | None = category_filters.narrative, print_doc=True):
    """Print annotations of both annotators and the doc for analysis"""

    annot1 = filter_func(doc.spans['annot1']) if filter_func is not None else doc.spans['annot1']
    annot2 = filter_func(doc.spans['annot2']) if filter_func is not None else doc.spans['annot2']

    print('annotator 1'.center(50, '-'), '|', 'annotator 2'.center(50, '-'))
    span_pairs(annot1, annot2)
    if print_doc: print('-' * 104, '\n', doc.text)


def storyline(annot: list[tokens.Span]):
    return '\n'.join(f'{span.label_}:\n{span!r}\n' for span in category_filters.storyline(annot))

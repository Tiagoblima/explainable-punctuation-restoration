"""Utility functions for comparing annotations from 2 annotators"""

from __future__ import annotations
from spacy import tokens
from typing import Callable


def quantity_concordance(annot1: list[tokens.Span], annot2: list[tokens.Span], filter_func: Callable = None) -> int:
    """Compare if these two annotations agree on the number of narrative elements"""
    if filter_func is not None:
        annot1 = filter_func(annot1)
        annot2 = filter_func(annot2)
    return len(annot1) == len(annot2)


def intersection(annot1: list[tokens.Span], annot2: list[tokens.Span]) -> list[tuple[tokens.Span, tokens.Span]]:
    """Naive intersection of spans
    ex:
    ```
        a = doc[2:5]
        b = doc[3:5]
        c = doc[5:6]
        (a, b) -> YES
        (b, b) -> YES
        (a, c) -> NO
    ```
    """
    
    def intersect(s1: tokens.Span, s2: tokens.Span) -> bool:
        """Detects if s1 intersects s2"""
        return s1.start in range(s2.start, s2.end) or s1.end-1 in range(s2.start, s2.end)

    return [
        (span1, span2)
        for span1 in annot1
        for span2 in annot2
        if intersect(span1, span2)
    ]


def smart_intersection(
    annot1: list[tokens.Span],
    annot2: list[tokens.Span]
) -> list[tuple[tokens.Span, tokens.Span]]:
    """Get intersection of spans without duplicates
    
    Duplicated annotation give preference to matching span the other annotator

    ex:
    ```
        a = doc[2:5]
        b = doc[3:5]
        c = doc[4:6]
        (a, b) -> NO   (because (b, b) is preferred)
        (b, b) -> YES
        (a, c) -> YES
    ```
    """
    result = intersection(annot1, annot2)

    # todos os spans que possuem um concordante
    has_match = ((a, b) for a,b in result if a.label == b.label)
    has_match = set(sum(has_match, start=()))

    return [
        (a,b) for a,b in result
        if a.label == b.label
        or (a not in has_match and b not in has_match)
    ]


def check_matching_labels(annot1: list[tokens.Span], annot2: list[tokens.Span], filter_func: Callable = None) -> list[bool]:
    """Check how much the annotators agree on the annotated intersection of spans
    
    ex: Labels bellow each other intersect

    annot1 -> | alpha | beta | charlie | 
    annot2 -> | alpha | beta | delta   | charlie 

    result -> [ True  , True , False   ]
    """
    if filter_func is not None:
        annot1 = filter_func(annot1)
        annot2 = filter_func(annot2)
    return [s1.label == s2.label for s1, s2 in smart_intersection(annot1, annot2)]


def merge(
    annot1: list[tokens.Span],
    annot2: list[tokens.Span],
    scorer: Callable[[tokens.Span], float] = len
) -> list[tokens.Span]:
    """Join two span annotations, merging spans with the same label.

    scorer: biggest score remains

    logic:
    - Select spans that overlap and have the same label for comparison (annotators saying the same thing)
    - Reduce selected spans, getting the largest of each (annotator that gives more detail)
    - Return reduced and not selected spans
    """
    # Selection
    agreed_pairs = [
        pair for pair in smart_intersection(annot1, annot2)
        if pair[0].label == pair[1].label
    ]

    if not agreed_pairs:
        return sorted(list(annot1) + list(annot2))

    # Unselected spans
    annot1_selected, annot2_selected = map(set, zip(*agreed_pairs))
    annot1_remaining = [a for a in annot1 if a not in annot1_selected]
    annot2_remaining = [a for a in annot2 if a not in annot2_selected]

    # Reduction
    annotations = [max(pair, key=scorer) for pair in agreed_pairs]

    # Concatenation
    return sorted(annotations + annot1_remaining + annot2_remaining)

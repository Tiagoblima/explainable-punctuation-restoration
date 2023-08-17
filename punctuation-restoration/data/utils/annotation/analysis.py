from __future__ import annotations
from itertools import chain
from typing import Callable
from spacy import tokens
import statistics as stats
from . import comparison


# Diante de todas as anotações de se intersectam, quantas são iguais? (percentual)
def intersection_agreement(docs: list[tokens.Doc], filter_func: Callable = None) -> float:
    """Percentage of overlaping spans that have the same label"""

    assert {'annot1', 'annot2'}.issubset(docs[0].spans.keys()), 'docs must have annot[1,2] keys for annotation comparison between annotators'

    concordance_per_doc = [
        comparison.check_matching_labels(doc.spans['annot1'], doc.spans['annot2'], filter_func=filter_func) for doc in docs]
    concordance_general = list(chain(*concordance_per_doc))
    try:
        return stats.mean(concordance_general)
    except stats.StatisticsError:
        raise Exception("All docs have empty annotations or the filter function cleaned all")
    


# Diante de totas as anotações quantas tem a mesma quantidade de elementos narrativos (percentual)
def quantity_agreement(docs: list[tokens.Doc], filter_func: Callable = None) -> float:
    """Percentage of agreement on the quantity of annotations on text"""
    
    assert {'annot1', 'annot2'}.issubset(docs[0].spans.keys()), 'docs must have annot[1,2] keys for annotation comparison between annotators'

    result = [comparison.quantity_concordance(doc.spans['annot1'], doc.spans['annot2'], filter_func=filter_func) for doc in docs]
    return stats.mean(result)

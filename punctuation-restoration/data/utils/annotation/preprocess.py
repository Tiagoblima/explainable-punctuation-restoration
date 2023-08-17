from __future__ import annotations
import re
from typing import Callable, Iterable
from typing_extensions import Literal
from spacy import tokens, training, util
from spellchecker import SpellChecker
from simpletransformers.ner import NERModel, NERArgs
from math import prod

import functools
# from mec_nlp.orthography import segmentation


def get_title(text: str) -> str:
    """Split title and text based on the `[T]` marker at the beginning of the first line

    obs: titles are valid only at the beginning of a text
    """
    match = re.search(r'(?<=^\[T\]).*\n+', text)

    if not match:
        return ''
    return match.group().strip()


def clean_text_tags(text: str, out_of_vocab='OOV') -> str:
    """Replace text tags keeping valid indexes"""
    # unknown char joined to word | Title markup | legacy tags
    text = re.sub(r'\[[SXx\?]\](?=\w+)|\[T\]|<\w>', '   ', text)
    # unknown word
    text = re.sub(r'\[[SXx\?]\]', out_of_vocab, text)
    # paragraph
    text = text.replace('[P]', '\t  ')
    return text


def get_words(doc: tokens.Doc) -> list[str]:
    return [tok.text for tok in doc]


def get_spaces(doc: tokens.Doc) -> list[bool]:
    return [bool(tok.whitespace_) for tok in doc]


def get_iob(doc: tokens.Doc) -> list[str]:
    return training.iob_utils.biluo_to_iob(training.iob_utils.doc_to_biluo_tags(doc))

def get_pos(doc: tokens.Doc) -> list[str]:
    return [tok.pos_ for tok in doc]

def get_sent_starts(doc: tokens.Doc) -> list[bool | int | None] | None:
    return [tok.is_sent_start for tok in doc] if doc.has_annotation('SENT_START') else None


def spans2ents(doc: tokens.Doc, label: str, spans_key: str = 'errors') -> tokens.Doc:
    new = doc.copy()
    new.spans.clear()
    ents = [
        tokens.Span(new, s.start, s.end, s.label_)
        for s in doc.spans[spans_key]
        if s.label_ == label
    ]
    new.set_ents(util.filter_spans(ents))
    return new


class AnnotationTracker:
    """Keep track of Doc annotations"""

    def __init__(self, spans_keys: Iterable[str] | str | None = None) -> None:
        self.spans_keys = (spans_keys,) if isinstance(spans_keys, str) else spans_keys


BadTokenChecker = Callable[[tokens.Token], bool]

class TokenRemover(AnnotationTracker):

    def __call__(self,
        doc: tokens.Doc,
        words: set[str] = {},
        when: BadTokenChecker = None,
        add_space: bool = False
    ) -> tokens.Doc:
        return self.remove(doc, words=words, when=when, add_space=add_space)

    def update_span_on_remove(self, span: tokens.Span, index: int) -> tokens.Span:
        start = (span.start -1) if index < span.start else span.start
        end = (span.end -1) if index < span.end else span.end
        return tokens.Span(span.doc, start, end, span.label)

    def remove(self,
        doc: tokens.Doc,
        words: set[str] = {},
        when: BadTokenChecker = None,
        add_space: bool = False
    ) -> tokens.Doc:
        """Remove words from doc

        Args:
            doc (tokens.Doc): Document
            words (set[str]): Set of words to remove
            when (BadTokenChecker): Function that receives a token and check if it is bad
            add_space (bool, optional): Add space to the left of each word removed. Defaults to False.

        Returns:
            tokens.Doc: A new doc without the `words`
        """
        found = set()
        if words:
            found.update(tok.i for tok in doc if tok.text in words)
        if when is not None:
            found.update(tok.i for tok in doc if when(tok))
        return self.remove_tokens(doc, indexes=found, add_space=add_space)

    def remove_tokens(self, doc: tokens.Doc, indexes: Iterable[int], add_space: bool = False) -> tokens.Doc:
        spans_keys = tuple(doc.spans.keys()) if self.spans_keys is None else self.spans_keys
        words = get_words(doc)
        spaces = get_spaces(doc)
        iob = get_iob(doc)
        pos = get_pos(doc)

        spans = doc.spans.copy()
        removed = 0
        indexes = set(indexes)
        for index in range(len(words)):
            if index in indexes:
                fixed_index = index - removed

                if add_space and fixed_index > 0:
                    spaces[fixed_index -1] = True

                words.pop(fixed_index)
                spaces.pop(fixed_index)
                iob.pop(fixed_index)
                pos.pop(fixed_index)

                for key in spans_keys:
                    update = (self.update_span_on_remove(span, fixed_index) for span in spans[key])
                    spans[key] = [span for span in update if len(span) > 0]
                removed += 1

        new = tokens.Doc(doc.vocab, words, spaces, ents=iob, pos=pos)
        new.spans.update(spans.copy(new))
        return new


class TokenReplacer(AnnotationTracker):
    def __call__(self, doc: tokens.Doc, replace: dict[str, str]) -> tokens.Doc:
        return self.replace_tokens(doc, replace)

    def replace_tokens(self, doc: tokens.Doc, replace: dict[str, str]) -> tokens.Doc:
        words = get_words(doc)
        spaces = get_spaces(doc)
        iob = get_iob(doc)
        pos = get_pos(doc)

        words = [replace[word] if word in replace else word for word in words]
        new = tokens.Doc(doc.vocab, words, spaces, ents=iob, pos=pos)
        new.spans.update(doc.spans.copy(new))
        return new


class TokenCorrector(TokenReplacer):
    def __init__(self, spans_keys: Iterable[str] | str | None = None, spellchecker: SpellChecker | None = None) -> None:
        self.spell = SpellChecker('pt', distance=2) if spellchecker is None else spellchecker
        super().__init__(spans_keys)

    def __call__(self, doc: tokens.Doc) -> tokens.Doc:
        return self.correction(doc)
    
    def fix_word(self, word: str) -> str | None:
        size = len(word)
        if size < 3:
            return None
        if size < 6:
            self.spell.distance = 1
            return self.spell.correction(word)
        else:
            self.spell.distance = 2
            return self.spell.correction(word)

    def correction(self, doc: tokens.Doc) -> tokens.Doc:
        replace = {
            tok.lower_: correction
            for tok in doc
            if tok.lower_ not in self.spell 
            and (correction := self.fix_word(tok.lower_)) is not None
        }
        return self.replace_tokens(doc, replace)


class HypersegmentationCorrector(AnnotationTracker):
    def __init__(self, spans_keys: Iterable[str] | str | None = None, spellchecker: SpellChecker | None = None, ngram = 2) -> None:
        self.ngram = ngram
        self.spell = SpellChecker('pt', distance=1) if spellchecker is None else spellchecker
        super().__init__(spans_keys)

    def __call__(self, doc: tokens.Doc) -> tokens.Doc:
        return self.correction(doc)
    
    def detect(self, span: tokens.Span) -> Literal[False] | str:
        """Detects and corrects a span of tokens in case of hypersegmentation

        Return: `correction if hypersegmented else False`
        """

        if not all(tok.is_alpha for tok in span) \
            or all(tok.lower_ in self.spell.word_frequency for tok in span):
            return False

        word = ''.join(tok.lower_ for tok in span)

        # se a concatenação estiver no vocabulário
        if word in self.spell:
            return word

        # se a concatenação precisava de uma pequena correção (distâcia 1)
        fixed = self.spell.correction(word)
        if fixed != word:
            return False if fixed is None else fixed

        # se mesmo concatenando e corrigindo, a palavra não fizer sentido,
        # então era apenas um erro e não uma hipersegmentação
        return False
    

    def update_span(self, span: tokens.Span, start: int, end: int) -> tokens.Span:
        """Update span considering that tokens on interval (`start`, `end`) will be joined in one token at `start`"""
        reduction = (end - start) - 1
        start = (span.start - reduction) if start < span.start else span.start
        end = (span.end - reduction) if start < span.start else span.end
        return tokens.Span(span.doc, start, end, span.label)

    def correction(self, doc: tokens.Doc) -> tokens.Doc:
        """Fix hypersegmentations keeping track of annotations"""
        spans_keys = tuple(doc.spans.keys()) if self.spans_keys is None else self.spans_keys
        words = get_words(doc)
        spaces = get_spaces(doc)
        iob = get_iob(doc)
        pos = get_pos(doc)

        removed = 0
        
        spans = doc.spans.copy()

        for curr in range(0, len(doc) - self.ngram + 1):
            span = doc[curr:curr+self.ngram]
            correction = self.detect(span)

            if correction is not False:
                start = span.start - removed
                end = span.end - removed
                removed += end - start - 1

                words[start:end] = [correction]

                spaces_fix = [spaces[end-1]]
                spaces[start:end] = spaces_fix

                iob_fix = [iob[end-1]]
                iob[start:end] = iob_fix

                pos_fix = [pos[end-1]]
                pos[start:end] = pos_fix


                for key in spans_keys:
                    update = (self.update_span(span, start, end) for span in spans[key])
                    spans[key] = [span for span in update if len(span) > 0]

        new = tokens.Doc(doc.vocab, words, spaces, ents=iob, pos=pos)
        new.spans.update(spans.copy(new))
        return new

class NorvigHypoSegmentaton:
    """Norvig's hiposegmentation algorithm

    adapted from https://norvig.com/ngrams/index.html MIT license
    """

    def __init__(self, dicio: SpellChecker) -> None:
        self.dicio = dicio

    def __call__(self, text: str) -> list[str]:
        return self.segment(text)

    @functools.lru_cache(100)
    def segment(self, text: str) -> list[str]:
        "Return a list of words that is the best segmentation of text."
        if not text:
            return []
        candidates = ([first]+self.segment(rem)
                      for first, rem in self.splits(text))
        return max(candidates, key=self.Pwords)

    def splits(self, text: str, max_len: int = 30) -> list[tuple[str, str]]:
        "Return a list of all possible (first, rem) pairs, len(first)<=L."
        return [(text[:i+1], text[i+1:]) for i in range(min(len(text), max_len))]

    def Pwords(self, words: list[str]) -> float:
        "The Naive Bayes probability of a sequence of words."
        return prod(self.dicio.word_usage_frequency(w) for w in words)


class HyposegmentationCorrector(AnnotationTracker):
    def __init__(self, spans_keys: Iterable[str] | str | None = None, spellchecker: SpellChecker | None = None) -> None:
        self.segment = NorvigHypoSegmentaton(SpellChecker('pt') if spellchecker is None else spellchecker)
        super().__init__(spans_keys)

    def __call__(self, doc: tokens.Doc) -> tokens.Doc:
        return self.correction(doc)
    
    def update_span(self, span: tokens.Span, index: int, size: int) -> tokens.Span:
        increment = size - 1
        start = (span.start + increment) if index < span.start else span.start
        end = (span.end + increment) if index < span.start else span.end
        return tokens.Span(span.doc, start, end, span.label)
        
    def correction(self, doc: tokens.Doc) -> tokens.Doc:
        spans_keys = tuple(doc.spans.keys()) if self.spans_keys is None else self.spans_keys
        words = get_words(doc)
        spaces = get_spaces(doc)
        iob = get_iob(doc)
        pos = get_pos(doc)

        dummy = tokens.Doc(doc.vocab, [' ']*(len(doc)*100))
        spans = doc.spans.copy(dummy)

        added = 0
        for tok in doc:
            correction = self.segment(tok.lower_)
            if len(correction) > 1:
                index = tok.i + added
                words[index:index+1] = correction
                spaces[index:index+1] = [True]*(len(correction)-1) + [bool(tok.whitespace_)]
                iob[index:index+1] = [iob[index]]*len(correction)
                pos[index:index+1] = [pos[index]]*len(correction)

                for key in spans_keys:
                    update = (self.update_span(span, index, len(correction)) for span in spans[key])
                    spans[key] = [span for span in update if len(span) > 0]

        new = tokens.Doc(doc.vocab, words, spaces, ents=iob, pos=pos)
        new.spans.update(spans.copy(new))
        return new


class PuntuationRestorer(AnnotationTracker):


    """ Essa classe implementa um restaurador de pontuação,
    capaz de receber um texto como entrada e produzir sua versão
    pontuada como saída. Exemplo de uso:

    >>> restorer = PunctuationRestoration()
    >>> restorer('...') # Saída é uma str com o texto pontuado
    """

    def __init__(self, model: NERModel | None = None, max_sequence_length: int = 512, spans_keys: Iterable[str] | str | None = None) -> None:

        if model is None:
            model_path = 'data/punctuation-recovery.v1/'
            model_args = NERArgs(
                labels_list=["O", "COMMA", "PERIOD", "QUESTION"],
                max_seq_length=max_sequence_length,
                use_multiprocessing=False,
                use_multiprocessing_for_evaluation=False,
                silent=True
            )
            model = NERModel("bert", model_path, args=model_args, use_cuda=False)

        self.model = model
        self.max_sequence_length = max_sequence_length
        self.marks = {
            'COMMA': ',',
            'PERIOD': '.',
            'QUESTION': '?'
        }
        super().__init__(spans_keys=spans_keys)

    def update_span(self, span: tokens.Span, index: int) -> tokens.Span:
        start = (span.start + 1) if index < span.start else span.start
        end = (span.end + 1) if index < span.start else span.end
        return tokens.Span(span.doc, start, end, span.label)

    def __call__(self, doc: tokens.Doc) -> tokens.Doc:
        return self.restore(doc)

    def restore(self, doc: tokens.Doc) -> tokens.Doc:
        spans_keys = tuple(doc.spans.keys()) if self.spans_keys is None else self.spans_keys

        words = get_words(doc)
        spaces = get_spaces(doc)
        iob = get_iob(doc)
        pos = get_pos(doc)

        result: list[dict[str, str]] = self.model.predict([words], split_on_space=False)[0][0]

        dummy = tokens.Doc(doc.vocab, [' ']*(len(doc)*100))
        spans = doc.spans.copy(dummy)

        added = 0
        for i, pair in enumerate(result):
            token, tag = tuple(pair.items())[0]
            if tag not in self.marks:
                continue
            punct = self.marks[tag]
            if token == punct:
                continue
            
            # adicionando pontuação
            added += 1
            index = i + added

            spaces.insert(index, True)
            spaces[index - 1] = False

            words.insert(index, punct)
            pos.insert(index, 'PUNCT')
            iob.insert(index, 'O')

            for key in spans_keys:
                update = (self.update_span(span, index) for span in spans[key])
                spans[key] = [span for span in update if len(span) > 0]

        new = tokens.Doc(doc.vocab, words, spaces, ents=iob, pos=pos)
        new.spans.update(spans.copy(new))
        return new

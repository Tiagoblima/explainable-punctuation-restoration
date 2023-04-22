"""Filtros de Qualidade Textual"""
from __future__ import annotations
from pathlib import Path
from spacy import tokens
from spellchecker import SpellChecker
import re


dictionary = Path('data/usp-spell-wordfreq.gz')
spell = SpellChecker('pt', local_dictionary=str(dictionary) if dictionary.exists() else None)


def remove_bad_tokens(doc):
    return remover.remove(doc, words={'OOV', '\n'}, when=lambda token: token.is_space, add_space=True)


# informações
def orthography(text: str) -> float:
    """Ortografia"""
    words = spell.split_words(text)
    unkown_words = spell.unknown(words)
    if not words: return False
    return len(unkown_words) / len(words)


def erasure(text: str) -> float:
    """Rasuras"""
    words = spell.split_words(text)
    if not words: return False
    return len(re.findall(r'\[[SXx\?]\]|(OOV)', text)) / len(words)


def num_lines(text: str) -> int:
    """Minimo de linhas"""
    return text.count('\n') + 1


def num_words(text: str) -> int:
    """Minimo de palavras"""
    return len(spell.split_words(text))


def viability(
        text: str,
        mispells: float = 0.2,
        erasures: float = 0.4,
        min_lines: int = 5,
        min_words: int = 80,
        resume: bool = True
) -> bool | list[bool]:
    """Testa 4 paramentros de viabilidade de correção de um texto
    
    mispells: percentual de erros ortográficos
    erasures: percentual de rasuras
    min_lines: número minimo de linhas
    min_words: número minimo de palavras
    resume: return result passed to `all` function, false returns list of booleans
    """
    result = [
        orthography(text) < mispells,
        erasure(text) < erasures,
        num_lines(text) >= min_lines,
        num_words(text) >= min_words,
    ]
    if resume:
        return all(result)
    return result


def docs_viability(docs: list[tokens.Doc], **kwargs) -> list[tokens.Doc]:
    """Filtro de viabilidade de correção
    
    kwargs: os mesmos de `viability`
    """
    return [doc for doc in docs if viability(doc.text, **kwargs)]

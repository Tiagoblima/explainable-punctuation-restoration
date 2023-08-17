"""Módulo de Filtros de Anotações

OBS: Nomes de Variaveis em português para manter a consistencia 
     e proximidade com os nomes dos rótulos usados
"""
from __future__ import annotations
from spacy import tokens
from functools import partial


class Category:
    """Categoria de Rótulos Anotados"""
    labels: frozenset

    def __new__(cls): ...  # impedindo instanciação

    def __init_subclass__(cls):
        cls.labels = frozenset(
            value for key, value in cls.__dict__.items()
            if key[0] != '_' and isinstance(value, str)
        )


class Enredo(Category):
    ORIENTAÇÃO = 'Orientação'
    COMPLICAÇÃO = 'Complicação'
    DESFECHO = 'Desfecho'


class Narrativa(Category):
    NARRADOR = 'Narrador'
    PERSONAGEM = 'Personagem'
    TEMPO = 'Organização temporal'
    LUGAR = 'Lugar/Espaço'
    AÇÃO = 'Ação'


class Pontuação(Category):
    ERRO = 'Erro de Pontuação'
    VÍRGULA = 'Erro de vírgula'


class Ortografia(Category):
    ERRO = 'grafia de palavras'
    SEGMENTATION = 'Desvios de hipersegmentação/ hipossegmentação'


class Morfologia(Category):
    PRONOMES = 'Erro pronomes pessoais'


class Estilística(Category):
    ORALIDADE = 'Presença de elementos da oralidade'


class Sintaxe(Category):
    CONCORDÂNCIA_NOMINAL_VERBAL = 'Erro concordância nominal/verbal'
    REGÊNCIA_NOMINAL_VERBAL = 'Erro regência nominal/verbal'
    ESTRUTURA = 'Períodos compostos mal estruturados sintaticamente'


class Semântica(Category):
    REPETIÇÃO = 'Emprego repetitivo de palavras'
    CONECTORES = 'Erros de conectores ou de palavras de referência'
    CONJUGAÇÃO = 'Incorreção na conjugação verbal'
    PALAVRA = 'Palavra semanticamente inadequada'


class Plágio(Category):
    PARÁFRASE = 'Paráfrase texto motivador'
    CÓPIA = 'Cópia texto motivador'


def filter_annotations(
        annotations: list[tuple[int, int, str]] | list[tokens.Span],
        labels: set[str]
) -> list[tuple[int, int, str]] | list[tokens.Span]:
    try:
        return [span for span in annotations if span.label_ in labels]
    except AttributeError:  # label_
        return [tuple(i) for i in annotations if i[2] in labels]


storyline = partial(filter_annotations, labels=Enredo.labels)
narrative = partial(filter_annotations, labels=Narrativa.labels)
punctuation = partial(filter_annotations, labels=Pontuação.labels)
orthography = partial(filter_annotations, labels=Ortografia.labels)
plagiarism = partial(filter_annotations, labels=Plágio.labels)
semantic = partial(filter_annotations, labels=Semântica.labels)
syntax = partial(filter_annotations, labels=Sintaxe.labels)
stylistics = partial(filter_annotations, labels=Estilística.labels)
morphology = partial(filter_annotations, labels=Morfologia.labels)

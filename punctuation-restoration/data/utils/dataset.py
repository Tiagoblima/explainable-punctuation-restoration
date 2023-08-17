from __future__ import annotations
from pathlib import Path
from typing import Callable
from typing_extensions import Literal

import srsly
import json
import re

from .annotation import preprocess
from . import annotation
from spacy import tokens
import spacy


nlp = spacy.blank("pt")


def convert(
    path: str | Path = "data",
    token_alignment: Literal["contract", "expand"] = "expand",
    preprocess_func: Callable[[str], str] = preprocess.clean_text_tags,
) -> list[tokens.Doc]:
    """Converte jsonl do doccano para o estilo de anotação do SpaCy e todos dos docs encontrados

    Arguments:
        path: path to the folder with `Semana*/` subfolders
        token_alignment: strategy to use when capturing a char span
            example:
                contract -> 'h[e was]' -> 'was'
                expand   -> 'h[e was]' -> 'he was'

        remove_title: remove title from text and update annotation spans
        preprocess: callable that clean the text *without* changing the size of text
    """

    path = Path(path)
    result = [p for p in path.glob("**/Anot*")]
    result.sort()

    colors = {
        item["text"]: item["backgroundColor"]
        for item in json.load((path / "tags.json").open())
    }

    docs = []
    for week_path in result:
        week = int(re.search(r"(?<=Semana)\d+", str(week_path)).group())

        # paths -> json generators -> list[json]
        jsonls = list(week_path.glob("anot*"))
        jsonls.sort()

        annotated_jsonl = [tuple(srsly.read_jsonl(path)) for path in jsonls]
        annotated_pairs = tuple(zip(*annotated_jsonl))

        for annotators in annotated_pairs:

            # verifica se os textos entre os anotadores são iguais
            assert all(
                annotators[0]["text"] == d["text"] for d in annotators
            ), "A sequencia de textos está diferente. por tanto não é possível comparar"

            # cria o documento pelo texto do primeiro anotador
            text = annotators[0]["text"]
            title = preprocess_func(preprocess.get_title(text))

            if preprocess_func:
                # this function should not change the size
                text = preprocess_func(text)

            doc = nlp.make_doc(text)
            doc.user_data["week"] = week
            doc.user_data["title"] = title
            doc.user_data["colors"] = colors

            for annotator_id, annotation in enumerate(annotators, start=1):
                # criando spans a partir dos indices
                spans = (
                    doc.char_span(*s, alignment_mode=token_alignment)
                    for s in annotation["label"]
                )

                # removendo indices que não foram puderam se alinhar a tokens do texto
                # cria uma entrada nos spans do doc com o número do anotador
                doc.spans["annot%d" % annotator_id] = [
                    s for s in spans if s is not None
                ]

            # texto só é válido se todos os anotadores anotarem algo
            if all(doc.spans.values()):
                docs.append(doc)
    return docs


def create(
    path: str = "data",
    token_alignment: Literal["contract", "expand"] = "expand",
    preprocess: Callable = preprocess.clean_text_tags,
    span_scorer: Callable[[tokens.Span], float] = len,
    span_key: str = "all",
    merge_key: str = "all",
    merge: bool = True,
) -> list[tokens.Doc]:
    """Create Dataset with the best spans of each annotator
    path: path to the folder with `Semana*/` subfolders
    token_alignment: strategy to use when capturing a char span

        example:
            contract -> 'h[e was]' -> 'was'\n
            expand   -> 'h[e was]' -> 'he was'


    preprocess: callable that clean the text *without* changing the size of text
    span_scorer: used to compare two spans biggest score remains
    """
    docs = convert(path, token_alignment, preprocess)

    if merge:
        annotation.merge(docs, output=merge_key, scorer=span_scorer)
    annotation.categorize(docs, span_key=span_key)
    return docs


def save(docs: list[tokens.Doc], path: str | Path) -> None:
    tokens.DocBin(store_user_data=True, docs=docs).to_disk(path)

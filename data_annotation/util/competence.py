from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

def preprocess(anotador1, anotador2):
    texts = anotador1.text
    concatenated = pd.concat(dict(text=texts, annot1=anotador1.label, annot2=anotador2.label), axis=1)

    # separando anotações delimitadas pelo '#'
    competences_lists = concatenated.iloc[:, 1:].applymap(lambda x: [int(comp.split()[-1]) for comp in x.split('#')], na_action='ignore')

    # verificando se todas as colunas possuem as 4 avaliações
    good_indexes = competences_lists.applymap(lambda x: len(x) == 4, na_action='ignore').all(axis=1)
    assert all(competences_lists.index == good_indexes.index)

    # mantendo apenas linhas com listas de tamanho 4 e removendo NaN
    competences_lists = competences_lists[good_indexes].dropna()

    # construindo dataframe composto
    competences = pd.concat({
        anot: pd.DataFrame(
            data = competences_lists[anot].array.tolist(),
            columns = ['comp%i' % i for i in range(1,5)],
            index = competences_lists.index
        ) for anot in competences_lists.columns
    }, axis=1)

    # adicionando coluna superior ao 'text'
    # texts = pd.concat(dict(raw=texts.to_frame()), axis=1)

    # unindo frames mantendo apenas textos que passaram no preprocessamento
    competences.insert(0, 'text', texts[competences.index])
    return competences


def preprocess_week(path: Path) -> pd.DataFrame:
    annot1 = pd.read_csv(path / 'Classes/anotador1.csv', index_col='id')
    annot2 = pd.read_csv(path / 'Classes/anotador2.csv', index_col='id')
    return preprocess(annot1, annot2)


def load_dataset(path: str = 'data', flat: bool = True):
    weeks = [path for path in Path(path).glob('Semana*') if path.is_dir()]
    weeks.sort()
    weeks = list(map(preprocess_week, weeks))
    df = pd.concat([week.assign(week=i) for i, week in enumerate(weeks, 1)])
    df = df[['text', 'week', 'annot1', 'annot2']]
    df = df.convert_dtypes(convert_integer=False)
    if flat:
        df.columns = df.columns.map('_'.join).str.strip('_')
    return df


@np.vectorize
def aprox(v1 ,v2):
    if v1 == v2:
        return 1.0
    """
    1 - (abs(5 - 1) - 1) / 3 == 0    # pior
    1 - (abs(5 - 4) - 1) / 3 == 1    # melhor
    """
    return 1 - (abs(v1 - v2) - 1) / 3

@np.vectorize
def vizinho(v1 ,v2):
    return float(abs(v1 - v2) < 2)

def most_concordant_joined(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Une avaliações de nível vizinho prevalecendo o maior nível
    e descarta as que não são avaliações vizinhas

    Por descartar redações, os resultados não são do mesmo tamanho
    
    ex: para cada competencia
        5 e 4 -> entra: 5
        1 e 2 -> entra: 2
        2 e 4 -> descarta
        1 e 5 -> descarta

    """
    results = {}
    is_neighbor_rows = vizinho(df.annot1, df.annot2).T.astype(bool)
    competence_names = ['comp%d' % i for i in range(1,5)]
    for neighbor, comp in zip(is_neighbor_rows, competence_names):
        columns = [['week', ''], ['annot1', comp], ['annot2', comp], ['text', '']]
        comp_df = df.loc[neighbor][columns].droplevel(1, 1)
        comp_df['level'] = comp_df[['annot1', 'annot2']].max(axis=1)
        results[comp] = comp_df[['week', 'level', 'text']]
    return results


def most_discrepant(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    results = {}
    is_neighbor_rows = vizinho(df.annot1, df.annot2).T.astype(bool)
    competence_names = ['comp%d' % i for i in range(1,5)]
    for neighbor, comp in zip(is_neighbor_rows, competence_names):
        columns = [['week', ''], ['annot1', comp], ['annot2', comp], ['text', '']]
        comp_df = df.loc[~neighbor][columns]
        comp_df.columns = 'week', 'annot1', 'annot2', 'text'
        results[comp] = comp_df
    return results

def most_concordant_joined(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    results = {}
    is_neighbor_rows = vizinho(df.annot1, df.annot2).T.astype(bool)
    competence_names = ['comp%d' % i for i in range(1,5)]
    for neighbor, comp in zip(is_neighbor_rows, competence_names):
        columns = [['week', ''], ['annot1', comp], ['annot2', comp], ['text', '']]
        comp_df = df.loc[neighbor][columns].droplevel(1, 1)
        comp_df['level'] = comp_df[['annot1', 'annot2']].max(axis=1)
        results[comp] = comp_df[['week', 'level', 'text']]
    return results
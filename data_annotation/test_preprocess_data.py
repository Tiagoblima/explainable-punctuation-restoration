import spacy
from spacy import tokens, displacy

from data_annotation.preprocess_data import corrector, hyperseg, hyposeg, remover
from data_annotation.util.show import log


def print_sents(doc):
    print(*doc.sents, sep=' 🔶 ')


def print_sent_starts(doc):
    for tok in doc:
        print(tok.is_sent_start, '\t-->', tok.text)
text = '''
texto para OOV testar errus, contidos em anotassões
OOV de trechos Ontem eu fui para Recife.
Eu goto de uva
Quando a menina gos tava de tinta e Marcos fes uma sur presa
Ontem eufoi pra casa.
'''

nlp = spacy.load('pt_core_news_md')
doc = nlp(text)
doc.spans['errors'] = [
    tokens.Span(doc, 5, 6, 'ortografia'),
    tokens.Span(doc, 9, 10, 'ortografia'),
    tokens.Span(doc, 22, 23, 'ortografia'),
    tokens.Span(doc, 29, 31, 'hiperseg'),
    tokens.Span(doc, 35, 36, 'ortografia'),
    tokens.Span(doc, 37, 39, 'hiperseg'),
    tokens.Span(doc, 41, 42, 'hiposeg'),
]
displacy.render(doc, 'span', options=dict(spans_key='errors'))

print_sents(doc)
displacy.render(doc, 'span', options=dict(spans_key='errors'))

log('REMOVENDO "OOV" e "\\n"')
res = remover.remove(doc, {'OOV', '\n'}, add_space=True)
displacy.render(res, 'span', options=dict(spans_key='errors'))

log('CORRIGINDO TEXTO')
res = corrector.correction(res)
displacy.render(res, 'span', options=dict(spans_key='errors'))

log('CORRIGINDO HYPERSEGMENTAÇÕES')
res = hyperseg.correction(res)
displacy.render(res, 'span', options=dict(spans_key='errors'))

log('CORRIGINDO HIPOSEGMENTAÇÕES')
res = hyposeg.correction(res)
displacy.render(res, 'span', options=dict(spans_key='errors'))
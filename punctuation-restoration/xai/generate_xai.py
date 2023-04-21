import requests
from tqdm import tqdm
from zipfile import ZipFile
import os
import shutil
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers_interpret import TokenClassificationExplainer
from tokenizers.processors import TemplateProcessing

model = AutoModelForTokenClassification.from_pretrained(BERT_LARGE)
tokenizer = AutoTokenizer.from_pretrained(BERT_LARGE)

ner_explainer = TokenClassificationExplainer(
    model,
    tokenizer,
)

word_attributions = ner_explainer(sample_text, ignored_labels=['O'])
word_attributions
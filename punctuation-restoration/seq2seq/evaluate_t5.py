from transformers import pipeline


def t5_transform_sentence(text):
    text = text.replace(' [I-COMMA]', ',')
    text = text.replace(' [I-PERIOD]', '.')
    text = text.replace('[Other]', '')
    text = text.replace('Recognize Entities: ', '')
    return text


def get_t5_pred_sentence(sentence: str, model: pipeline):
    gen_text = model(sentence)[0]['generated_text']

    return t5_transform_sentence(gen_text)


def t5labels2text(text):
    text = text.replace(' [I-COMMA]', ',')
    text = text.replace(' [I-PERIOD]', '.')
    text = text.replace(' [Other]', '')
    text = text.replace('Recognize Entities: ', '')
    return text

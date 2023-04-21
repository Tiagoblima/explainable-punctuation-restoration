from transformers import pipeline


def bert_transform_sentences(text_, groups):
    new_text_list = list(text_)

    shift = 0

    for out in groups:
        punkt = '.' if out['entity_group'] == 'PERIOD' else ','
        if out['end'] + shift < len(new_text_list) + 1:
            new_text_list.insert(out['end'] + shift, punkt)
            shift += 1

    return ''.join(new_text_list)


def get_bert_pred_sentence(sentence: str, model: pipeline):
    groups = model(sentence)

    new_text = bert_transform_sentences(sentence, groups)
    return new_text
import spacy


def read_corpus_file(corpus_file, split_char=','):
    with open(corpus_file, encoding='utf-8') as file_:
        lines_ = file_.readlines()
    data_ = []
    words_ = []
    tags_ = []
    previous_id_sent = -1
    for line in lines_[1:]:
        line = line.replace('\n', '')
        if line == '':
            continue
        fragments = line.split(split_char)
        current_id_sent = int(fragments[0])
        if previous_id_sent != -1 and previous_id_sent != current_id_sent:
            data_.append((words_, tags_))
            words_ = []
            tags_ = []
        words_.append(fragments[2])
        tags_.append(fragments[1])
        previous_id_sent = current_id_sent
    data_.append((words_, tags_))
    return data_


def data_preprocessing(data):
    nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner', 'lemmatizer', 'textcat'])
    preprocessed_data = []
    for d in data:
        sentence = ' '.join(d[0])
        doc = nlp(sentence)
        pos_tags = [t.pos_ for t in doc]
        preprocessed_data.append((d[0], pos_tags, d[1]))
    return preprocessed_data


"""
    Implementation based on
        https://www.aitimejournal.com/@akshay.chavan/complete-tutorial-on-named-entity-recognition-ner-using-python-and-keras
"""


def extract_sent_features(sentence):
    return [extract_features(sentence, i) for i in range(len(sentence))]


def extract_labels(sentence):
    return [label for _, _, label in sentence]


def extract_features(sentence, i):
    word = sentence[i][0]
    postag = sentence[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'word.islower()': word.islower(),
        'word[0].isupper()': word[0].isupper(),
        'word[0].islower()': word[0].islower(),
        'not word[0].isalnum()': not word[0].isalnum(),
        'not word.isalnum()': not word.isalnum(),
        'word.isalpha()': word.isalpha()
    }
    if i > 0:
        word1 = sentence[i - 1][0]
        postag1 = sentence[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:word.islower()': word1.islower()
        })
    else:
        features['BOS'] = True
    if i > 1:
        word1 = sentence[i - 2][0]
        postag1 = sentence[i - 2][1]
        features.update({
            '-2:word.lower()': word1.lower(),
            '-2:word.istitle()': word1.istitle(),
            '-2:word.isupper()': word1.isupper(),
            '-2:postag': postag1,
            '-2:postag[:2]': postag1[:2],
            '-2:word.islower()': word1.islower()
        })
    if i < len(sentence) - 1:
        word1 = sentence[i + 1][0]
        postag1 = sentence[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:word.islower()': word1.islower()
        })
    else:
        features['EOS'] = True
    if i < len(sentence) - 2:
        word1 = sentence[i + 2][0]
        postag1 = sentence[i + 2][1]
        features.update({
            '+2:word.lower()': word1.lower(),
            '+2:word.istitle()': word1.istitle(),
            '+2:word.isupper()': word1.isupper(),
            '+2:postag': postag1,
            '+2:postag[:2]': postag1[:2],
            '+2:word.islower()': word1.islower()
        })
    return features


def convert_data(data):
    sentences = []
    for d in data:
        sentences.append(list(zip(d[0], d[1], d[2])))
    x_data = [extract_sent_features(s) for s in sentences]
    y_data = [extract_labels(s) for s in sentences]
    return x_data, y_data


def dump_report(rep_dict, output_file):
    if not rep_dict:
        print('\nReport is empty!')
        return
    with open(output_file, 'w') as out_file:
        out_file.write('label,precision,recall,f1-score\n')
        for label in rep_dict.keys():
            if label != 'accuracy':
                out_file.write(f"{label},{rep_dict[label]['precision']},{rep_dict[label]['recall']},"
                               f"{rep_dict[label]['f1-score']}\n")
            else:
                out_file.write(f"{label},,,{rep_dict[label]}\n")

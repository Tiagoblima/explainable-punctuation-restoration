import os


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


def save_data(list_data, save_path, name_file):

    string_data = ''

    for data in list_data:
        for w, t in zip(data[0], data[1]):
            string_data += f'{w}\t{t}\n'
        string_data += '\n'

    new_file_path = os.path.join(save_path, name_file)

    with open(new_file_path, 'w', encoding='utf-8') as file:
        file.write(string_data)


if __name__ == '__main__':

    # corpus_name = 'obras'
    corpus_name = 'tedtalk2012'

    train_file = f'./data/tedtalk2012/train.csv'
    dev_file = f'./data/tedtalk2012/dev.csv'
    test_file = f'./data/tedtalk2012/test.csv'

    new_files_dir = f'./data/{corpus_name}'

    os.makedirs(new_files_dir, exist_ok=True)

    test_data = read_corpus_file(test_file, split_char=',')
    dev_data = read_corpus_file(dev_file, split_char=',')
    train_data = read_corpus_file(train_file, split_char=',')

    save_data(train_data, new_files_dir, name_file='train.csv')
    save_data(dev_data, new_files_dir, name_file='dev.csv')
    save_data(test_data, new_files_dir, name_file='test.csv')

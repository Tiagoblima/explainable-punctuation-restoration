def generate_test_file(test_results_file, new_test_file):
    with open(new_test_file, 'w+', encoding='utf8') as new_file:
        with open(test_results_file, 'r', encoding='utf8') as file:
            for line in file:
                if line != '\n':
                    line = line.strip()
                    spliter = line.split(' ')
                    token = spliter[0]
                    tag_1 = spliter[1]
                    tag_2 = spliter[2]
                    new_file.write(str(token) + ' ' + str(tag_1) +
                                   ' ' + str(tag_2) + '\n')
                else:
                    new_file.write(line)

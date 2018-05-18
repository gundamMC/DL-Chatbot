import re  # regex


def cornell_cleanup(sentence):
    # clean up html tags
    sentence = re.sub(r'<.*?>', '', sentence)
    # clean up \n and \r
    return sentence.replace('\n', '').replace('\r', '')


def load_cornell(path_conversations, path_lines):
    movie_lines = {}
    lines_file = open(path_lines, 'r', encoding="iso-8859-1")
    for line in lines_file:
        line = line.split(" +++$+++ ")
        line_number = line[0]
        character = line[1]
        movie = line[2]
        sentence = line[-1]

        if movie not in movie_lines:
            movie_lines[movie] = {}

        movie_lines[movie][line_number] = (character, sentence)

    conversations = []
    conversations_file = open(path_conversations, 'r', encoding="iso-8859-1")
    for line in conversations_file:
        line = line.split(" +++$+++ ")
        movie = line[2]
        line_numbers = []
        for num in line[3][1:-2].split(", "):
            line_numbers.append(num[1:-1])

        # Not used since the cornell data set already placed
        # the lines of the same character together
        #
        # lines = []
        #
        # tmp = []
        #
        # teacher = movie_lines[movie][line_numbers[0]][0]
        # # teacher is the one that speaks first
        # was_teacher = True
        #
        # for num in line_numbers:
        #
        #     line = movie_lines[movie][num]
        #     if line[0] == teacher:
        #         if not was_teacher:  # was the bot
        #             lines.append([True, tmp])  # append previous conversation and mark as "is bot"
        #             tmp = []
        #         tmp.append(cornell_cleanup(line[1]))
        #         was_teacher = True
        #     else:  # bot speaking
        #         if was_teacher:  # was teacher
        #             lines.append([False, tmp])  # append previous conversation and mark "is not bot"
        #             tmp = []
        #         tmp.append(cornell_cleanup(line[1]))
        #         was_teacher = False
        #
        # if len(tmp) > 0:
        #     lines.append([not was_teacher, tmp])  # append the last response (not b/c of the inverse)
        #
        # conversations.append(lines)

        for i in range(len(line_numbers) - 1):
            input_line = movie_lines[movie][line_numbers[i]][1]
            output_line = movie_lines[movie][line_numbers[i + 1]][1]
            conversations.append([cornell_cleanup(input_line), cornell_cleanup(output_line)])

    return conversations


def split_sentence(sentence):
    # collect independent words
    return re.findall(r"[\w']+|[.,!?;]", sentence)


def split_data(data):
    result = []
    for conversation in data:
        result.append([split_sentence(conversation[0]), split_sentence(conversation[1])])
    return result


def sentence_to_index(sentence, word_to_index):
    result = []
    for word in sentence:
        if word in word_to_index:
            result.append(word_to_index[word])
        else:
            result.append(word_to_index["<UNK>"])
    # max sequence length of 50
    if len(result) < 49:  # last one will always be eos
        result.extend([word_to_index["<PAD>"]] * (49 - len(result)))
        result.append(word_to_index["<EOS>"])
        return result
    else:
        result = result[:50]
        result.append(word_to_index["<EOS>"])
        return result


def data_to_index(data, word_to_index):
    result = []
    for conversation in data:
        result.append([sentence_to_index(conversation[0], word_to_index),
                       sentence_to_index(conversation[1], word_to_index)])
    return result

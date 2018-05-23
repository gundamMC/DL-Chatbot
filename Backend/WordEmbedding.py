from Utils import print_message
import numpy as np


words = None
words_to_index = None
embeddings = None

start = 0
end = 0


def create_embedding(glove_path):
    global words
    words = []
    global words_to_index
    words_to_index = {}
    global embeddings
    embeddings = []
    print_message("Loading word vector")

    # add special tokens
    zeros = np.zeros((4, 54))

    words_to_index["<UNK>"] = 0
    words.append("<UNK>")
    zeros[0, 50] = 1

    words_to_index["<PAD>"] = 1
    words.append("<PAD>")
    zeros[1, 51] = 1

    words_to_index["<EOS>"] = 2
    global end
    end = 2
    words.append("<EOS>")
    zeros[2, 52] = 1

    words_to_index["<GO>"] = 3
    global start
    start = 3
    words.append("<GO>")
    zeros[3, 53] = 1

    f = open(glove_path, 'r', encoding='utf8')
    for index, line in enumerate(f):
        split_line = line.split(' ')
        word = split_line[0]
        embedding = [float(val) for val in split_line[1:]]
        words.append(word)
        embedding.extend([0, 0, 0, 0])
        embeddings.append(embedding)
        words_to_index[word] = index + 4  # 4 special tokens

    embeddings = np.array(embeddings)

    embeddings = np.vstack((zeros, embeddings))

    assert embeddings[end, 52] == 1
    assert embeddings[start, 53] == 1

    print_message("Word vector loaded")

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
    f = open(glove_path, 'r', encoding='utf8')
    for index, line in enumerate(f):
        split_line = line.split(' ')
        word = split_line[0]
        embedding = [float(val) for val in split_line[1:]]
        words.append(word)
        embedding.extend([0, 0, 0, 0])
        embeddings.append(embedding)
        words_to_index[word] = index

    embeddings = np.array(embeddings)

    zeros = np.zeros((4, 29))

    words_to_index["<UNK>"] = len(words)
    words.append("<UNK>")
    zeros[0, 25] = 1

    words_to_index["<PAD>"] = len(words)
    words.append("<PAD>")
    zeros[1, 26] = 1

    words_to_index["<EOS>"] = len(words)
    global end
    end = len(words)
    words.append("<EOS>")
    zeros[2, 27] = 1

    words_to_index["<GO>"] = len(words)
    global start
    start = len(words)
    words.append("<GO>")
    zeros[3, 28] = 1

    embeddings = np.vstack((embeddings, zeros))

    print_message("Word vector loaded")

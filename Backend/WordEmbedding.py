from Utils import print_message
import numpy as np


words = None
words_to_index = None
embeddings = None


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
        embedding.extend([0, 0, 0])
        embeddings.append(embedding)
        words_to_index[word] = index

    words_to_index["<UNK>"] = len(words)
    words.append("<UNK>")
    unk = [[0] * 50, [1], [0], [0]]
    embeddings.append(unk)

    words_to_index["<PAD>"] = len(words)
    words.append("<PAD>")
    pad = [[0] * 50, [0], [1], [0]]
    embeddings.append(pad)

    words_to_index["<EOS>"] = len(words)
    words.append("<EOS>")
    eos = [[0] * 50, [0], [0], [1]]
    embeddings.append(eos)

    embeddings = np.array(embeddings)

    print_message("Word vector loaded")

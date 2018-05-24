import numpy as np


words = None
words_to_index = None
embeddings = None

start = 0
end = 0


def create_embedding(glove_path, save_embedding=True):
    global words
    words = []
    global words_to_index
    words_to_index = {}
    global embeddings
    embeddings = []

    print("Loading word vector")

    words_to_index["<UNK>"] = 0
    words.append("<UNK>")

    words_to_index["<PAD>"] = 1
    words.append("<PAD>")

    words_to_index["<EOS>"] = 2
    global end
    end = 2
    words.append("<EOS>")

    words_to_index["<GO>"] = 3
    global start
    start = 3
    words.append("<GO>")

    f = open(glove_path, 'r', encoding='utf8')
    for index, line in enumerate(f):
        split_line = line.split(' ')
        word = split_line[0]
        if save_embedding:
            embedding = [float(val) for val in split_line[1:]]
            embedding.extend([0, 0, 0, 0])
            embeddings.append(embedding)
        words.append(word)
        words_to_index[word] = index + 4  # 4 special tokens

    if save_embedding:
        # add special tokens
        zeros = np.zeros((4, 54))
        zeros[0] = np.random.rand(1, 54)
        zeros[1, 51] = 1
        zeros[2, 52] = 0
        zeros[3] = np.ones((1, 54))

        embeddings = np.array(embeddings)
        embeddings = np.vstack((zeros, embeddings))

    print("Word vector loaded")

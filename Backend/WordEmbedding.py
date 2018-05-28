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

    words_to_index["<GO>"] = 2
    global start
    start = 2
    words.append("<GO>")

    f = open(glove_path, 'r', encoding='utf8')
    index = 3
    for line in f:

        if index > 25000:
            break

        split_line = line.split(' ')
        word = split_line[0]
        if save_embedding:
            embedding = [float(val) for val in split_line[1:]]
            # embedding.extend([0, 0, 0, 0])
            embeddings.append(embedding)
        words.append(word)
        words_to_index[word] = index  # 3 special tokens before
        index += 1

    words_to_index["<EOS>"] = len(words)
    global end
    end = len(words)
    words.append("<EOS>")

    if save_embedding:
        # add special tokens
        zeros = np.random.rand(3, 50)
        # zeros = np.zeros((4, 50))
        # zeros[0] = np.random.rand(1, 50)
        # zeros[1, 51] = 1
        # zeros[2, 52] = 0
        # zeros[3] = np.ones((1, 50))

        embeddings = np.array(embeddings)
        embeddings = np.vstack((zeros, embeddings, np.random.rand(1, 50)))

    print("Word vector loaded")

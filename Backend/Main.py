import ParseData
import WordEmbedding


test = ParseData.load_cornell("path to movie_conversations.txt", "path to movie_lines.txt")

test = ParseData.split_data(test)

WordEmbedding.create_embedding("path to glove")

test = ParseData.data_to_index(test, WordEmbedding.words_to_index)

print(test[0])

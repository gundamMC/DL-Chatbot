import ParseData
import WordEmbedding


question, response = ParseData.load_cornell("path to movie_conversations.txt", "path to movie_lines.txt")

question, question_length = ParseData.split_data(question)

WordEmbedding.create_embedding("path to glove")

test, test_length = ParseData.data_to_index(question, WordEmbedding.words_to_index)

print(test[0])

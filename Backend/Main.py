import ParseData
import WordEmbedding
from Network import ChatbotNetwork
import numpy as np


question, response = ParseData.load_cornell("path to movie_conversations.txt", "path to movie_lines.txt")
question = ParseData.split_data(question)
response = ParseData.split_data(response)

# TODO: Use a smaller word vector to conserve memory
WordEmbedding.create_embedding("path to glove.twitter.27B.25d.txt")

question_index, question_length = ParseData.data_to_index(question, WordEmbedding.words_to_index)
response_index, response_length = ParseData.data_to_index(response, WordEmbedding.words_to_index)

question_index = np.array(question_index)
print(question_index.shape)
question_length = np.array(question_length)
print(question_length.shape)
response_index = np.array(response_index)
print(response_index.shape)
response_length = np.array(response_length)
print(response_length.shape)

print(WordEmbedding.embeddings[question_index[0]].shape)

network = ChatbotNetwork()
network.train(question_index, question_length, response_index, response_length)

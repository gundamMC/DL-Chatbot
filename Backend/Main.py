import ParseData
import WordEmbedding
from Network import ChatbotNetwork
import numpy as np
import os

# Force Tensorflow to use CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

question, response = ParseData.load_cornell(".\\Data\\movie_conversations.txt", ".\\Data\\movie_lines.txt")
question = ParseData.split_data(question)
response = ParseData.split_data(response)

WordEmbedding.create_embedding(".\\Data\\glove.6B.50d.txt")

question_index, question_length = ParseData.data_to_index(question, WordEmbedding.words_to_index)
response_index, response_length = ParseData.data_to_index(response, WordEmbedding.words_to_index)

question_index = np.array(question_index[:256])
print(question_index.shape)
question_length = np.array(question_length[:256])
print(question_length.shape)
response_index = np.array(response_index[:256])
print(response_index.shape)
response_length = np.array(response_length[:256])
print(response_length.shape)

network = ChatbotNetwork()
network.train(question_index, question_length, response_index, response_length, display_step=1)

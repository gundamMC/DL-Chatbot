import ParseData
import WordEmbedding
from Network import ChatbotNetwork


question, response = ParseData.load_cornell("path to movie_conversations.txt", "path to movie_lines.txt")
question, question_length = ParseData.split_data(question)
response, response_length = ParseData.split_data(response)

for i in range(len(response)):
    response[i].insert(0, "<GO>")
    response[i].append("<EOS>")
    response_length[i] += 2

WordEmbedding.create_embedding("path to glove.twitter.27B.50d.txt")

question_index = ParseData.data_to_index(question, WordEmbedding.words_to_index)
response_index = ParseData.data_to_index(response, WordEmbedding.words_to_index)

network = ChatbotNetwork()
network.train(question_index, question_length, response_index, response_length)

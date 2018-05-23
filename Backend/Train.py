import ParseData
import WordEmbedding
from Network import ChatbotNetwork
import numpy as np

# Force Tensorflow to use CPU
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

question, response = ParseData.load_cornell(".\\Data\\movie_conversations.txt", ".\\Data\\movie_lines.txt")
question = ParseData.split_data(question)
response = ParseData.split_data(response)

WordEmbedding.create_embedding(".\\Data\\glove.6B.50d.txt")

question_index, question_length = ParseData.data_to_index(question, WordEmbedding.words_to_index)
response_index, response_length = ParseData.data_to_index(response, WordEmbedding.words_to_index)

question_index = np.array(question_index[:256])
question_length = np.array(question_length[:256])
response_index = np.array(response_index[:256])
response_length = np.array(response_length[:256])

network = ChatbotNetwork()

while True:
    user_input = input("Train epochs: ")
    if user_input == "exit":
        break

    if user_input == "save":
        network.save()

    try:
        epochs = int(user_input)
    except ValueError:
        print("Integer only")
        continue

    network.train(question_index, question_length, response_index, response_length,
                  epochs=epochs, display_step=epochs/10)

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

new = True

if os.path.isfile("./model/checkpoint") and \
        input("Create new network or restore from model? (type 'restore' to restore, else create new): ") == "restore":
    new = False

WordEmbedding.create_embedding(".\\Data\\glove.6B.50d.txt", save_embedding=new)

question_index, question_length = ParseData.data_to_index(question, WordEmbedding.words_to_index)
response_index, response_length = ParseData.data_to_index(response, WordEmbedding.words_to_index)

question_index = np.array(question_index[:2048])
question_length = np.array(question_length[:2048])
response_index = np.array(response_index[:2048])
response_length = np.array(response_length[:2048])


network = ChatbotNetwork(restore=not new)


while True:
    user_input = input("Train epochs: ")
    if user_input == "exit":
        break

    if user_input == "save":
        network.save()
        continue

    try:
        epochs = int(user_input)
    except ValueError:
        print("Integer only")
        continue

    network.train(question_index, question_length, response_index, response_length,
                  epochs=epochs, display_step=epochs/10)

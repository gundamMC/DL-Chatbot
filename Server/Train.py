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

WordEmbedding.create_embedding(".\\Data\\glove.twitter.27B.100d.txt", save_embedding=new)

start_index = 0
end_index = start_index + 81960

question_index, response_index, question_length, response_length = \
    ParseData.data_to_index(question[start_index:end_index], response[start_index:end_index], WordEmbedding.words_to_index)

question_index = np.array(question_index)
question_length = np.array(question_length)
response_index = np.array(response_index)
response_length = np.array(response_length)

network = ChatbotNetwork(restore=not new)
if new:
    # Free memory
    WordEmbedding.embeddings = None

while True:
    user_input = input("Train epochs: ")
    if user_input == "exit":
        break

    if user_input == "save":
        network.save()
        continue

    if user_input == "continue":
        while True:
            network.train(question_index, question_length, response_index, response_length, epochs=2)
            network.save()
            print("Done")

    if user_input.startswith('predict'):
        input_x, x_length, _ = ParseData.sentence_to_index(ParseData.split_sentence(user_input.replace("predict ", "", 1)), WordEmbedding.words_to_index)
        result = network.predict([input_x], [x_length])
        print(result)
        continue

    try:
        epochs = int(user_input)
    except ValueError:
        print("Integer only")
        continue

    network.train(question_index, question_length, response_index, response_length,
                  epochs=epochs, display_step=int(epochs/10))

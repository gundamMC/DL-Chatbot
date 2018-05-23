import socket
import Utils
import WordEmbedding
import ParseData
from Network import ChatbotNetwork


# initialize network with pre-trained model
WordEmbedding.create_embedding(".\\Data\\glove.6B.50d.txt", save_embedding=False)
network = ChatbotNetwork(restore=True)


s = socket.socket()
host = "localhost"
port = 8089
s.bind((host, port))
s.listen(1)

while True:
    conn, addr = s.accept()

    print('New connection from %s:%d' % (addr[0], addr[1]))

    Utils.set_socket(conn)

    while True:
        try:
            data = conn.recv(1024)
            data = data.decode("utf-8").strip()
            if not data or data == "":
                break
            elif data == 'exit':
                conn.close()
                break
            else:
                # predict the result
                input_x, x_length = ParseData.data_to_index(ParseData.split_data([data.replace("input: ", '', 1)]),
                                                            WordEmbedding.words_to_index)
                result = network.predict(input_x, x_length)
                Utils.print_message(result)

        except socket.timeout:
            print('Connection timed out')
            break

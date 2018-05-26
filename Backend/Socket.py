import socket
import Utils
import WordEmbedding
import ParseData
from Network import ChatbotNetwork
from itertools import groupby


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

                if "; " in data:
                    data = data.split("; ")
                    start_token = data[0].lower()
                    if start_token not in WordEmbedding.words_to_index:
                        start_token = "<UNK>"
                    data = data[1]
                else:
                    start_token = "<GO>"  # default start token of <GO>
                input_x, x_length = ParseData.data_to_index(ParseData.split_data([data]),
                                                            WordEmbedding.words_to_index)
                result = start_token + ' ' + network.predict(input_x, x_length, start_token=start_token)
                # remove consecutive duplicates
                # https://stackoverflow.com/questions/5738901/removing-elements-that-have-consecutive-duplicates-in-python
                result = [x[0] for x in groupby(result.split(' '))]
                tmp = ""
                for word in result:
                    tmp += word + " "
                Utils.print_message(tmp)

        except socket.timeout:
            print('Connection timed out')
            break

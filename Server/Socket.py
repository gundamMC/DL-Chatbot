import socket
import Utils
import WordEmbedding
import ParseData
from Network import ChatbotNetwork
from itertools import groupby


# initialize network with pre-trained model
WordEmbedding.create_embedding(".\\Data\\glove.twitter.27B.100d.txt", save_embedding=False)
network = ChatbotNetwork(restore=True)

print("Network loaded")

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
                input_x, x_length, _ = ParseData.sentence_to_index(ParseData.split_sentence(data),
                                                                   WordEmbedding.words_to_index)
                result = network.predict(input_x, x_length, start_token='<GO>')
                # remove consecutive duplicates
                # https://stackoverflow.com/questions/5738901/removing-elements-that-have-consecutive-duplicates-in-python
                result = [x[0] for x in groupby(result.split(' '))][:-2]

                print(result)

                tmp = ""
                for word in result:
                    tmp += word + " "
                Utils.print_message(tmp)

        except socket.timeout:
            print('Connection timed out')
            continue

        except socket.error:
            print('Connection error')
            continue

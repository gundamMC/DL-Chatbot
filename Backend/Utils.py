socket = None


def set_socket(input_socket):
    global socket
    socket = input_socket


def print_message(message):
    if socket is not None:
        socket.send(message.encode("UTF-8"))
    else:
        print(message)

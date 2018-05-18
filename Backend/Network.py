import tensorflow as tf


class ChatbotNetwork:

    def __init__(self, learning_rate=0.01, batch_size=256):
        # hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Network hyperparameters
        self.n_vector = 53  # glove 50 + 3 special tokens
        self.max_seuqence = 50
        self.n_hidden = 128

        # Tensorflow placeholders
        self.x = tf.placeholder("float", [None, self.max_seuqence])
        self.y = tf.placeholder("float", [None, self.max_seuqence])

        # Network parameters
        self.cell_encode = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
        self.cell_decode = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)

    def setTrainingData(self, data):
        pass

    def network(self, mode="train"):
        pass

    def train(self, epochs=500, display_step=50):
        pass

    def predict(self, sentence):
        pass

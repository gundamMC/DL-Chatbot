import tensorflow as tf


class ChatbotNetwork:

    def __init__(self, learning_rate=0.01, batch_size=256, vector_count=20000, vector_length=53):
        # hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Network hyperparameters
        self.n_vector = vector_length  # glove 50 + 3 special tokens
        self.vector_count = vector_count
        self.max_seuqence = 50
        self.n_hidden = 128

        # Tensorflow placeholders
        self.x = tf.placeholder("float", [None, self.max_seuqence])
        self.x_length = tf.placeholder("float", [None])
        self.y = tf.placeholder("float", [None, self.max_seuqence])
        self.y_length = tf.placeholder("float", [None])

        # Network parameters
        self.cell_encode = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
        self.cell_decode = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)

    def set_training_data(self, data):
        pass

    def network(self, mode="train"):
        encoder_out, encoder_states = tf.nn.dynamic_rnn(
            self.cell_encode,
            inputs=self.x,
            dtype=tf.float32,
            sequence_length=self.x_length,
            swap_memory=True)

        if mode == "train":
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=self.y,
                sequence_length=self.y_length
            )
        else:
            helper = None

        decoder = tf.contrib.seq2seq.BasicDecoder(
            self.cell_decode,
            helper,
            encoder_states,
            output_layer=None  # projection layer
        )

        outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=False)

        return outputs

    def train(self, epochs=500, display_step=50):
        pass

    def predict(self, sentence):
        pass

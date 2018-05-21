import tensorflow as tf
import numpy as np
import WordEmbedding


class ChatbotNetwork:

    def __init__(self, learning_rate=0.01, batch_size=256, vector_count=20000, vector_length=54):
        # hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Network hyperparameters
        self.n_vector = vector_length  # glove 50 + 3 special tokens
        self.vector_count = vector_count
        self.max_sequence = 50
        self.n_hidden = 128

        # Tensorflow placeholders
        self.x = tf.placeholder(tf.int32, [None, self.max_sequence])
        self.x_length = tf.placeholder(tf.int32, [None])
        self.y = tf.placeholder("float", [None, self.max_sequence])
        self.y_length = tf.placeholder(tf.int32, [None])
        self.word_embedding = tf.Variable(tf.constant(0.0, shape=WordEmbedding.embeddings.shape), trainable=False)

        # Network parameters
        self.cell_encode = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
        self.cell_decode = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)

        # Optimization
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y, logits=self.network())
        self.cost = (tf.reduce_sum(crossent) / self.batch_size)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        # Tensorflow initialization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        # Word embedding
        embedding_placeholder = tf.placeholder(tf.float32, shape=WordEmbedding.embeddings.shape)
        self.sess.run(self.word_embedding.assign(embedding_placeholder),
                      feed_dict={embedding_placeholder: WordEmbedding.embeddings})

    def network(self, mode="train"):
        embedded_x = tf.nn.embedding_lookup(self.word_embedding, self.x)

        print(embedded_x.shape)

        encoder_out, encoder_states = tf.nn.dynamic_rnn(
            self.cell_encode,
            inputs=embedded_x,
            dtype=tf.float32,
            sequence_length=self.x_length,
            swap_memory=True)

        projection_layer = tf.layers.Dense(self.vector_count)

        if mode == "train":
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=self.y,
                sequence_length=self.y_length
            )

            decoder = tf.contrib.seq2seq.BasicDecoder(
                self.cell_decode,
                helper,
                encoder_states,
                output_layer=projection_layer
            )

        else:

            # Replicate encoder infos beam_width times
            decoder_initial_state = tf.contrib.seq2seq.tile_batch(
                encoder_states, multiplier=3)

            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=self.cell_decode,
                        embedding=self.word_embedding,
                        start_tokens=tf.tile(WordEmbedding.start, [self.x.shape[0]]),
                        end_token=WordEmbedding.end,
                        initial_state=decoder_initial_state,
                        beam_width=3,
                        output_layer=projection_layer,
                        length_penalty_weight=0.0
            )

        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=False)

        logits = outputs.rnn_output

        return logits

    def train(self, train_x, x_length, train_y, y_length, epochs=500, display_step=50):
        for epoch in range(epochs+1):
            mini_batches_x, mini_batches_x_length, mini_batches_y, mini_batches_y_length \
                = self.random_mini_batches([train_x, x_length, train_y, y_length], self.batch_size)

            for batch in range(len(mini_batches_x)):
                batch_x = mini_batches_x[batch]
                batch_x_length = mini_batches_x_length[batch]
                batch_y = mini_batches_y[batch]
                batch_y_length = mini_batches_y_length[batch]

                self.sess.run(self.train_op, feed_dict={
                    self.x: batch_x,
                    self.x_length: batch_x_length,
                    self.y: batch_y,
                    self.y_length: batch_y_length
                })

                if epoch % display_step == 0:
                    cost_value = self.sess.run(self.cost, feed_dict={
                                    self.x: batch_x,
                                    self.x_length: batch_x_length,
                                    self.y: batch_y,
                                    self.y_length: batch_y_length
                                })

                    print("epoch:", epoch, "- (", batch, "/", self.batch_size, ") -", cost_value)

    def predict(self, sentence):
        # TODO
        pass

    @staticmethod
    def random_mini_batches(data_lists, mini_batch_size):
        m = data_lists[0].shape[0]
        mini_batches = []

        # shuffle data
        permutation = list(np.random.permutation(m))
        shuffled = []
        for data in data_lists:
            shuffled.append(data[permutation])

        mini_batch_number = int(m / float(mini_batch_size))

        # split into mini batches
        for data in shuffled:
            tmp = []
            for batch in range(0, mini_batch_number):
                tmp.append(data[batch * mini_batch_size: (batch + 1) * mini_batch_size])

            if m % mini_batch_size != 0:
                tmp.append(data[mini_batch_number * mini_batch_size:])

            mini_batches.append(tmp)

        return mini_batches

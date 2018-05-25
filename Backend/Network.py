import tensorflow as tf
import numpy as np
import WordEmbedding
import ParseData


class ChatbotNetwork:

    def __init__(self, learning_rate=0.00005, batch_size=32, restore=False):
        # hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Network hyperparameters
        self.n_vector = 50
        self.vector_count = len(WordEmbedding.words)
        self.max_sequence = 25
        self.n_hidden = 64

        # Tensorflow placeholders
        self.x = tf.placeholder(tf.int32, [None, self.max_sequence])
        self.x_length = tf.placeholder(tf.int32, [None])
        self.y = tf.placeholder(tf.int32, [None, self.max_sequence])
        self.y_length = tf.placeholder(tf.int32, [None])
        with tf.device('/cpu:0'):
            self.word_embedding = tf.Variable(tf.constant(0.0, shape=(self.vector_count, self.n_vector)), trainable=False)

        # Network parameters
        self.cell_encode_fw = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
        self.cell_encode_bw = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
        self.cell_decode = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
        self.projection_layer = tf.layers.Dense(self.vector_count, use_bias=False)

        # Optimization
        mask = tf.sequence_mask(self.y_length, maxlen=self.max_sequence, dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(logits=self.network(), targets=self.y, weights=mask)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        # Tensorflow initialization
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # Saver object which will save/restore all the variables
        self.saver = tf.train.Saver()

        if restore:
            self.saver.restore(self.sess, './model/model.ckpt')
        else:
            # get new word embedding
            with tf.device('/cpu:0'):
                embedding_placeholder = tf.placeholder(tf.float32, shape=WordEmbedding.embeddings.shape)
                self.sess.run(self.word_embedding.assign(embedding_placeholder),
                              feed_dict={embedding_placeholder: WordEmbedding.embeddings})

    def network(self, mode="train"):
        with tf.device('/cpu:0'):
            embedded_x = tf.nn.embedding_lookup(self.word_embedding, self.x)

        encoder_out, encoder_states = tf.nn.bidirectional_dynamic_rnn(
            self.cell_encode_fw,
            self.cell_encode_bw,
            inputs=embedded_x,
            dtype=tf.float32,
            sequence_length=self.x_length,
            swap_memory=True)

        encoder_states = encoder_states[-1]

        if mode == "train":

            with tf.device('/cpu:0'):
                embedded_y = tf.nn.embedding_lookup(self.word_embedding, self.y)

            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=embedded_y,
                sequence_length=tf.sign(self.y_length) * self.max_sequence
            )

            decoder = tf.contrib.seq2seq.BasicDecoder(
                self.cell_decode,
                helper,
                encoder_states,
                output_layer=self.projection_layer
            )

            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=self.max_sequence, swap_memory=True)

            return outputs.rnn_output

        else:

            # Beam search
            decoder_initial_state = tf.contrib.seq2seq.tile_batch(
                encoder_states, multiplier=3)

            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=self.cell_decode,
                        embedding=self.word_embedding,
                        start_tokens=tf.tile(tf.constant([WordEmbedding.words_to_index["<PAD>"]], dtype=tf.int32), [tf.shape(self.x)[0]]),
                        end_token=WordEmbedding.end,
                        initial_state=decoder_initial_state,
                        beam_width=3,
                        output_layer=self.projection_layer,
                        length_penalty_weight=1.0
            )

            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=self.max_sequence)

            return outputs.predicted_ids

    def train(self, train_x, x_length, train_y, y_length, epochs=500, display_step=10):
        for epoch in range(epochs+1):
            if epoch % 10 == 0:
                with tf.device('/cpu:0'):
                    # test 1
                    test_output = self.sess.run(self.network(mode="infer")[0],
                                                feed_dict={
                                                    self.x: train_x[0].reshape((1, self.max_sequence)),
                                                    self.x_length: x_length[0].reshape((1,))
                                                })

                    result = ""
                    for i in test_output:
                        if len(i) == 3:
                            result = result + WordEmbedding.words[round(i[0])] + "(" + str(i[0]) + ")" + " "
                        else:
                            break
                    print(result)

                    result = ""
                    for i in train_y[0, :y_length[0]]:
                        result = result + WordEmbedding.words[i] + "(" + str(i) + ")" + " "
                    print(result)

                    # test 2
                    test, test_length = ParseData.data_to_index([["how", "are", "you", "?"]],
                                                                WordEmbedding.words_to_index)
                    test_output = self.sess.run(self.network(mode="infer")[0],
                                                feed_dict={
                                                    self.x: np.array(test),
                                                    self.x_length: np.array(test_length)
                                                })

                    result = ""
                    for i in test_output:
                        if len(i) == 3:
                            result = result + WordEmbedding.words[round(i[0])] + "(" + str(i[0]) + ")" + " "
                        else:
                            break
                    print(result)

                    # end

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

                    print("epoch:", epoch, "- (", batch, "/", len(mini_batches_x), ") -", cost_value)

    def predict(self, sentence, sentence_length):
        test_output = self.sess.run(self.network(mode="infer")[0],
                                    feed_dict={
                                        self.x: np.array(sentence),
                                        self.x_length: np.array(sentence_length)
                                    })
        result = ""
        for i in test_output:
            result = result + WordEmbedding.words[round(i[0])] + " "
        return result

    def save(self):
        self.saver.save(self.sess, './model/model.ckpt')

    @staticmethod
    def random_mini_batches(data_lists, mini_batch_size):
        m = data_lists[0].shape[0]
        mini_batches = []

        # shuffle data
        permutation = list(np.random.permutation(m))
        shuffled = []
        for i in range(len(data_lists)):
            shuffled.append(data_lists[i][permutation])

        mini_batch_number = int(m / float(mini_batch_size))

        # split into mini batches
        for i in range(len(shuffled)):
            tmp = []
            for batch in range(0, mini_batch_number):
                tmp.append(data_lists[i][batch * mini_batch_size: (batch + 1) * mini_batch_size])

            if m % mini_batch_size != 0:
                tmp.append(data_lists[i][mini_batch_number * mini_batch_size:])

            mini_batches.append(tmp)

        return mini_batches

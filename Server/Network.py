import tensorflow as tf
import numpy as np
import WordEmbedding
import ParseData


class ChatbotNetwork:

    def __init__(self, learning_rate=0.0003, batch_size=8, restore=False):
        # hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Network hyperparameters
        self.n_vector = 100
        self.vector_count = len(WordEmbedding.words)
        self.max_sequence = 20
        self.n_hidden = 512
        self.n_layer = 2

        # Tensorflow placeholders
        self.x = tf.placeholder(tf.int32, [None, self.max_sequence])
        self.x_length = tf.placeholder(tf.int32, [None])
        self.y = tf.placeholder(tf.int32, [None, self.max_sequence])
        self.y_length = tf.placeholder(tf.int32, [None])
        self.y_target = tf.placeholder(tf.int32, [None, self.max_sequence])
        self.word_embedding = tf.Variable(tf.constant(0.0, shape=(self.vector_count, self.n_vector)), trainable=False)

        # Network parameters
        def get_gru_cell():
            return tf.contrib.rnn.GRUCell(self.n_hidden)
        self.cell_encode = tf.contrib.rnn.MultiRNNCell([get_gru_cell() for _ in range(self.n_layer)])
        self.cell_decode = tf.contrib.rnn.MultiRNNCell([get_gru_cell() for _ in range(self.n_layer)])
        self.projection_layer = tf.layers.Dense(self.vector_count)

        # Optimization
        dynamic_max_sequence = tf.reduce_max(self.y_length)
        mask = tf.sequence_mask(self.y_length, maxlen=dynamic_max_sequence, dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(self.network(),
                                                     self.y_target[:, :dynamic_max_sequence],
                                                     weights=mask)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        # Tensorflow initialization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        # Saver object which will save/restore all the variables
        self.saver = tf.train.Saver()

        if restore:
            self.saver.restore(self.sess, './model/model.ckpt')
        else:
            # get new word embedding
            embedding_placeholder = tf.placeholder(tf.float32, shape=WordEmbedding.embeddings.shape)
            self.sess.run(self.word_embedding.assign(embedding_placeholder),
                          feed_dict={embedding_placeholder: WordEmbedding.embeddings})

    def network(self, mode="train", start_token="<GO>"):
        embedded_x = tf.nn.embedding_lookup(self.word_embedding, self.x)

        _, encoder_state = tf.nn.dynamic_rnn(
            self.cell_encode,
            inputs=embedded_x,
            dtype=tf.float32,
            sequence_length=self.x_length)

        if mode == "train":

            embedded_y = tf.nn.embedding_lookup(self.word_embedding, self.y)

            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=embedded_y,
                sequence_length=self.y_length
            )

            decoder = tf.contrib.seq2seq.BasicDecoder(
                self.cell_decode,
                helper,
                encoder_state,
                output_layer=self.projection_layer
            )

            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=self.max_sequence)

            return outputs.rnn_output

        else:
            # Beam search
            decoder_initial_state = tf.contrib.seq2seq.tile_batch(
                encoder_state, multiplier=3)

            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=self.cell_decode,
                        embedding=self.word_embedding,
                        start_tokens=tf.tile(tf.constant([WordEmbedding.words_to_index[start_token]], dtype=tf.int32), [tf.shape(self.x)[0]]),
                        end_token=WordEmbedding.words_to_index["<EOS>"],
                        initial_state=decoder_initial_state,
                        beam_width=3,
                        output_layer=self.projection_layer,
                        length_penalty_weight=0.0
            )

            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=self.max_sequence)

            return tf.transpose(outputs.predicted_ids, perm=[0, 2, 1])  # [batch size, beam width, sequence length]

    def train(self, train_x, x_length, train_y, y_length, epochs=500, display_step=10):

        # stripping the first <GO> token and padding a <EOS> at the end
        y_target = np.concatenate([
            train_y[:, 1:],
            np.full([train_y.shape[0], 1], WordEmbedding.words_to_index['<EOS>'])
        ], axis=1)

        for epoch in range(epochs):

            mini_batches_x, mini_batches_x_length, mini_batches_y, mini_batches_y_length, mini_batches_y_target \
                = self.random_mini_batches([train_x, x_length, train_y, y_length, y_target], self.batch_size)

            for batch in range(len(mini_batches_x)):
                batch_x = mini_batches_x[batch]
                batch_x_length = mini_batches_x_length[batch]
                batch_y = mini_batches_y[batch]
                batch_y_length = mini_batches_y_length[batch]
                batch_y_target = mini_batches_y_target[batch]

                if (display_step == 0 or epoch % display_step == 0) and batch % 100 == 0:
                    _, cost_value = self.sess.run([self.train_op, self.cost], feed_dict={
                                        self.x: batch_x,
                                        self.x_length: batch_x_length,
                                        self.y: batch_y,
                                        self.y_length: batch_y_length,
                                        self.y_target: batch_y_target
                                    })

                    print("epoch:", epoch, "- (", batch, "/", len(mini_batches_x), ") -", cost_value)

                else:
                    self.sess.run(self.train_op, feed_dict={
                        self.x: batch_x,
                        self.x_length: batch_x_length,
                        self.y: batch_y,
                        self.y_length: batch_y_length,
                        self.y_target: batch_y_target
                    })

    def predict(self, sentence, sentence_length, start_token="<GO>"):
        test_output = self.sess.run(self.network(mode="infer", start_token=start_token)[0],
                                    feed_dict={
                                        self.x: np.array(sentence).reshape(-1, self.max_sequence),
                                        self.x_length: np.array(sentence_length).reshape(-1,)
                                    })
        result = ""
        for i in range(len(test_output[0])):
            result = result + WordEmbedding.words[int(test_output[0][i])] + " "
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

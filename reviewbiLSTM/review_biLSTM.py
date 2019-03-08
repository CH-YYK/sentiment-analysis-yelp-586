import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import fully_connected

class reviewLSTM(object):

    def __init__(self, review_length, sequence_length, word_vector, rnn_size, category_length, category_vector):
        # basic properties
        self.embedding_size = word_vector.shape[1]
        self.max_review_length = review_length
        self.max_words_length = sequence_length
        self.hidden_units = rnn_size
        self.embedding_cat_size = category_vector.shape[-1]

        # placeholders
        self.input_x = tf.placeholder(tf.int32, shape=[None, review_length, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, 5], name='label_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # category
        self.input_category = tf.placeholder(tf.int32, shape=[None, category_length], name='category')


        # embedding
        with tf.name_scope("embedding_word"):
            embedding_W = tf.get_variable("embedding_Matrix", initializer=tf.constant(word_vector, dtype=tf.float32),
                                          trainable=True)
            # [batch_size, num_reviews, sequence_length, words_embedding]
            self.embedding_words = tf.nn.embedding_lookup(embedding_W, self.input_x, name="embedded_words")

        # category_embedding
        with tf.name_scope('category_embeddings'):
            W_cat = tf.get_variable('W_cat', initializer=tf.constant(category_vector, dtype=tf.float32),
                                    trainable=True)
            self.embedded_cat = tf.nn.embedding_lookup(W_cat, self.input_category, name='embedded_cat')
        self.embedded_cat = tf.reduce_mean(self.embedded_cat, axis=1, keepdims=False)
        self.embedded_cat_exp = tf.concat([self.embedded_cat for i in range(self.max_review_length*self.max_words_length)], axis=-1)
        self.embedded_cat_exp = tf.reshape(self.embedded_cat_exp, [-1, self.max_words_length, category_vector.shape[-1]])

        # LSTM for each reviews: [batch_size, sequence_length, words_embedding]
        self.reviews_reshape = tf.reshape(self.embedding_words, shape=[-1, sequence_length, self.embedding_size],
                                          name='reshaped')

        with tf.name_scope("LSTM_review_embedding"):
            lstm_cell_fw = rnn.BasicLSTMCell(rnn_size, name="fw_cell")
            lstm_cell_fw = rnn.DropoutWrapper(lstm_cell_fw, output_keep_prob=self.keep_prob)

            lstm_cell_bw = rnn.BasicLSTMCell(rnn_size, name="bw_cell")
            lstm_cell_bw = rnn.DropoutWrapper(lstm_cell_bw, output_keep_prob=self.keep_prob)

            (fw_output, bw_output), (_, _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw,
                                                                             cell_bw=lstm_cell_bw,
                                                                             inputs=self.reviews_reshape,
                                                                             dtype=tf.float32)

            # :[batch_size, max_length_words, hidden_units*2]
            self.review_outputs = tf.concat([fw_output, bw_output], axis=2)
            self.review_embedding = self.Attention2_layer(self.review_outputs, name='words')

        self.flatten_output = tf.reshape(self.review_embedding, shape=[-1, review_length, self.hidden_units * 2])
        self.flatten_output = self.Attention_Layer(self.flatten_output, name='reviews')
        # self.flatten_output = tf.concat([tf.reduce_mean(self.embedded_cat, axis=-2), self.flatten_output], axis=-1)

        # batch normalization
        # self.flatten_output = tf.layers.batch_normalization(self.flatten_output)

        l2_loss = tf.constant(0.0)

        # output
        with tf.name_scope("output"):
            output_w = tf.get_variable("output_weight", shape=[self.flatten_output.shape[1].value, 5],
                                       initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            output_bias = tf.get_variable("output_bias", initializer=tf.constant([0.1]*5))

            l2_loss += tf.nn.l2_loss(output_w)
            l2_loss += tf.nn.l2_loss(output_bias)

            self.scores = tf.nn.xw_plus_b(self.flatten_output, output_w, output_bias, name="ouput_layer")
            self.output = tf.argmax(self.scores, axis=1)

        # loss and accuracy
        with tf.name_scope("loss_accuracy"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + 0.12 * l2_loss
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.output, tf.argmax(self.input_y, axis=1)), 'float'))

    def Attention_Layer(self, input_, name):
        """
        :param input_: output from last Bi-RNN
        :param name: For 'word' encoder or 'sentence' encoder
        :return: vector encoded
        """
        shape = input_.shape
        with tf.name_scope('%s_Attention' % name):
            weight = tf.get_variable("AttentionWeight_%s" % name,
                                     initializer=tf.truncated_normal([shape[-1].value], mean=0, stddev=0.01),
                                     dtype=tf.float32)
            # :[*batch_size, length_*, hidden_units * 2]
            h = fully_connected(input_, shape[-1].value, tf.nn.tanh)

            # :[*batch_size, length_*, 1]
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(weight, h), keepdims=True, axis=2), dim=1)

            # :[*batch_size, hidden_units*2]
            return tf.reduce_sum(tf.multiply(input_, alpha), axis=1)

    def Attention2_layer(self, input_, name):
        shape = input_.shape
        with tf.name_scope('%s_tag_attention' % name):
            weight = tf.get_variable("tag_attentionweight_%s" % name,
                                     initializer=tf.truncated_normal([shape[-1].value + self.embedding_cat_size]),
                                     dtype=tf.float32)
            # input: [batch*max_review_size, max_length_words, embedding_size]
            # categories: [batch, max_length_category, category_embedding]
            h = tf.concat([input_, self.embedded_cat_exp], axis=-1)
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(weight, h), keepdims=True, axis=-1), dim=1)
            return tf.reduce_sum(tf.multiply(input_, alpha), axis=1)


if __name__ == '__main__':
    test = reviewLSTM(5, 100, np.zeros((1000, 200), dtype=np.float32), 128, 1000, np.zeros((1000, 200)))

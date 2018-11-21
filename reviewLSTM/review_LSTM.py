import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

class reviewLSTM(object):

    def __init__(self, review_length, sequence_length, word_vector, rnn_size):
        self.input_x = tf.placeholder(tf.int32, shape=[None, review_length, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, 5], name='label_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # embedding
        with tf.name_scope("embedding_layer"):
            embedding_W = tf.get_variable("embedding_Matrix", initializer=tf.constant(word_vector, dtype=tf.float32),
                                          trainable=False)
            # [batch_size, num_reviews, sequence_length, words_embedding]
            self.embedding_words = tf.nn.embedding_lookup(embedding_W, self.input_x, name="embedded_words")

        # LSTM for each reviews: [batch_size, sequence_length, words_embedding]
        reviews_List = [review[:, 0, :, :] for review in tf.split(self.embedding_words, self.input_x.shape[1].value, axis=1)]
        flatten_output = []
        for index, review in enumerate(reviews_List):
            with tf.name_scope("LSTM_review_%d" % (index+1)):
                lstm_cell = rnn.BasicLSTMCell(rnn_size, name="cell_%d" % (index + 1))
                lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
                output, states = tf.nn.dynamic_rnn(lstm_cell, review, dtype=tf.float32)
                # [batch_size, LSTM_size]
                flatten_output.append(states.h)
        self.flatten_output = tf.concat(flatten_output, axis=1)

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
            self.loss = tf.reduce_mean(losses) + 0.06 * l2_loss
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.output, tf.argmax(self.input_y, axis=1)), 'float'))

if __name__ == '__main__':
    test = reviewLSTM(5, 100, np.zeros((1000, 200), dtype=np.float32), 128)
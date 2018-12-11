import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class TextRNN(object):
    def __init__(self, sequence_length, embedding_size, rnn_size, word_vector):

        # placeholders
        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, 5], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='Dropout_keep_prob')
        self.real_length = tf.placeholder(tf.int32, name='real_length')

        # embedding
        with tf.name_scope('embeddings'):
            W = tf.get_variable('W', initializer=tf.constant(word_vector, dtype=tf.float32), trainable=False)
            # [batch_size, sequence_length, embedding_sixze]
            self.embedded_char = tf.nn.embedding_lookup(W, self.input_x, name='embedded_chars')
            # self.embedded_char_expanded = tf.expand_dims(self.embedded_char, axis=-1)

        # RNN - LSTM
        with tf.name_scope("rnn_sequence"):
            lstm_cell = rnn.BasicLSTMCell(rnn_size)
            lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
            self.output_, self.state = tf.nn.dynamic_rnn(lstm_cell, self.embedded_char,
                                                        sequence_length=self.real_length, dtype=tf.float32)

        # output layer
        l2_loss = tf.constant(0.0)
        with tf.name_scope("output"):
            W = tf.get_variable('output_W', shape=[rnn_size, 5],
                                initializer=tf.contrib.layers.xavier_initializer())

            b = tf.Variable(initial_value=tf.constant([0.01] * 5), name='output_bias')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.state.h, W, b, name='scores')
            self.output = tf.argmax(self.scores, axis=1, name='output')

        # compute loss and accuracy
        with tf.name_scope('loss_and_accuracy'):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(loss) + 0.01 * l2_loss
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.output, tf.argmax(self.input_y, axis=1)), "float"))





import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np

class TextCNN(object):
    """
    input_x: placeholder, sequence of integers that represent sentences
    input_y: placeholder, a one-hot vector that represent label
    """

    def __init__(self, sequence_length, embedding_size, word_vector, filter_sizes, num_filters,
                 category_length, category_vector):
        # basic properties:
        self.sequence_lenth = sequence_length
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        # basic properties for categories
        self.embedding_cate_size = category_vector.shape[-1]
        self.category_length = category_length

        # define placeholders
        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, 5], name='label_y')
        self.input_category = tf.placeholder(tf.int32, shape=[None, category_length], name='category')
        self.keep_prob = tf.placeholder(tf.float32, name='Dropout_keep_prob')

        # word embeddings
        with tf.name_scope('text_embeddings'):
            W_word = tf.get_variable('W_word', initializer=tf.constant(word_vector, dtype=tf.float32), trainable=True)

            self.embedded_char = tf.nn.embedding_lookup(W_word, self.input_x, name='embedded_chars')
            self.embedded_char_expanded = tf.expand_dims(self.embedded_char, axis=-1)

        # category_embedding
        with tf.name_scope('category_embeddings'):
            W_cat = tf.get_variable('W_cat', initializer=tf.constant(category_vector, dtype=tf.float32), trainable=True)
            self.embedded_cat = tf.nn.embedding_lookup(W_cat, self.input_category, name='embedded_cat')
            self.embedded_cat_exp = tf.expand_dims(self.embedded_cat, dim=-1)

        # cnn with multi-filters and pooling
        pooling_output = []
        for i, filter_size in enumerate(filter_sizes):
            pool = self.cnn(input=self.embedded_char_expanded, filter_size=filter_size, index=i)
            pooling_output.append(pool)

        # cnn on category and pooling
        pooling_category = []
        filter_sizes_ = [2, 3]
        for i, filter_size in enumerate(filter_sizes_):
            pooling_category.append(self.cnn_category(input_=self.embedded_cat_exp, index=i, filter_size=filter_size))
        pooling_output += pooling_category

        # flatten all pooling output
        self.pool = tf.concat(pooling_output, axis=-1)
        total_num_neorons = num_filters * (len(filter_sizes) + len(filter_sizes_))

        # category mapping
        # self.embedded_cat_ = tf.reduce_mean(self.embedded_cat, axis=-1, name="weighed_categories_embedding")

        self.pool = tf.reshape(self.pool, shape=[-1, total_num_neorons])
        # self.pool = tf.concat([self.embedded_cat_, self.pool], axis=-1)

        # add dropout:
        with tf.name_scope('Dropout'):
            self.drop_out = tf.nn.dropout(self.pool, keep_prob=self.keep_prob)

        with tf.name_scope('fully_connnected'):
            self.full_connect = fully_connected(self.drop_out, num_outputs=500, activation_fn=tf.nn.relu)
            # self.full_connect_2 = fully_connected(self.full_connect, num_outputs=250, activation_fn=tf.nn.relu)

        l2_loss = tf.constant(0.0)
        # get output
        with tf.name_scope('output'):
            W = tf.get_variable('output_W', shape=[500, 5],
                                initializer=tf.contrib.layers.xavier_initializer())

            b = tf.Variable(initial_value=tf.constant([0.01] * 5), name='output_bias')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.full_connect, W, b, name='scores')
            self.output = tf.argmax(self.scores, axis=1, name='output')

        # compute loss and accuracy
        with tf.name_scope('loss_and_accuracy'):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(loss) + 0.5 * l2_loss
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.output, tf.argmax(self.input_y, axis=1)), "float"))

    def cnn(self, input, filter_size, index):
        with tf.name_scope('cnn_maxpool_%s' % index):
            filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
            W = tf.get_variable(name="cnn_Weight_{}_{}".format(filter_size, index),
                                initializer=tf.truncated_normal(shape=filter_shape, stddev=0.1))

            b = tf.get_variable(name="cnn_bias_{}_{}".format(filter_size, index),
                                initializer=tf.constant(0.1, shape=[self.num_filters]))
            # convolutional layer
            conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding="VALID", name='conv')

            # add bias and apply non-linearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), 'relu')

            # apply max_pooling
            return tf.nn.max_pool(h, ksize=[1, self.sequence_lenth - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                  padding='VALID')
    def cnn_category(self, input_, index, filter_size):
        with tf.name_scope('cnn_maxpool_category_%s' % index):
            filter_shape = [filter_size, self.embedding_cate_size, 1, self.num_filters]
            W_cate = tf.get_variable(name='cnn_weight_cate_%s' % filter_size,
                                     initializer=tf.truncated_normal(shape=filter_shape, stddev=0.1))
            b_cate = tf.get_variable(name='cnn_bias_cate_%s' % filter_size,
                                     initializer=tf.constant(0.1, shape=[self.num_filters]))
            conv = tf.nn.conv2d(input_, W_cate, strides=[1, 1, 1, 1], padding="VALID", name='conv_cate_%s' % index)
            h_cate = tf.nn.relu(tf.nn.bias_add(conv, b_cate), 'relu_cate_%s' % index)

            return tf.nn.max_pool(h_cate, ksize=[1, self.category_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                  padding='VALID')

if __name__ == '__main__':
    test = TextCNN(100, 100, np.zeros((100, 100)), [3, 4, 5, 6], 128, 10, np.zeros((10, 100)))
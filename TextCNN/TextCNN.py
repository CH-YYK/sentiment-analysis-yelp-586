import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

class TextCNN(object):
    """
    input_x: placeholder, sequence of integers that represent sentences
    input_y: placeholder, a one-hot vector that represent label
    """

    def __init__(self, sequence_length, embedding_size, word_vector, filter_sizes, num_filters):
        # basic properties:
        self.sequence_lenth = sequence_length
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        # define placeholders
        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, 5], name='label_y')
        self.keep_prob = tf.placeholder(tf.float32, name='Dropout_keep_prob')

        # word embeddings
        with tf.name_scope('embeddings'):
            W = tf.get_variable('W', initializer=tf.constant(word_vector, dtype=tf.float32), trainable=False)

            self.embedded_char = tf.nn.embedding_lookup(W, self.input_x, name='embedded_chars')
            self.embedded_char_expanded = tf.expand_dims(self.embedded_char, axis=-1)

        # cnn with multi-filters and pooling
        pooling_output = []
        for i, filter_size in enumerate(filter_sizes):
            pool = self.cnn(input=self.embedded_char_expanded, filter_size=filter_size, index=i)
            pooling_output.append(pool)

        # flatten all pooling output
        self.pool = tf.concat(pooling_output, axis=-1)
        total_num_neorons = num_filters * len(filter_sizes)

        self.pool = tf.reshape(self.pool, shape=[-1, total_num_neorons])

        # add dropout:
        with tf.name_scope('Dropout'):
            self.drop_out = tf.nn.dropout(self.pool, keep_prob=self.keep_prob)

        with tf.name_scope('fully_connnected'):
            self.full_connect = fully_connected(self.drop_out, num_outputs=500, activation_fn=tf.nn.relu)

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
            self.loss = tf.reduce_mean(loss) + 0.05 * l2_loss
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.output, tf.argmax(self.input_y, axis=1)), "float"))

    def cnn(self, input, filter_size, index):
        with tf.name_scope('cnn_maxpool_%s' % index):
            filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
            W = tf.get_variable(name="cnn_Weight_%s" % filter_size,
                                initializer=tf.truncated_normal(shape=filter_shape, stddev=0.1))

            b = tf.get_variable(name="cnn_bias_%s" % filter_size,
                                initializer=tf.constant(0.1, shape=[self.num_filters]))
            # convolutional layer
            conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding="VALID", name='conv')

            # add bias and apply non-linearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), 'relu')

            # apply max_pooling
            return tf.nn.max_pool(h, ksize=[1, self.sequence_lenth - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                  padding='VALID')

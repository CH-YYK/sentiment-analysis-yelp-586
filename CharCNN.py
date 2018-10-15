import tensorflow as tf

class CharCNN(object):

    def __init__(self, sequence_length, conv_config, fc_config, char_vector, num_classes=5):
        l2_loss = 0
        # define Placeholders
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length], name='input_x')
        self.label_y = tf.placeholder(dtype=tf.int32, shape=[None, num_classes], name='label')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')

        #
        with tf.name_scope("character_embedding"):
            W = tf.Variable(dtype=tf.float32, initial_value=char_vector,trainable=False,
                            name="character_embedding")
            self.input_ = tf.nn.embedding_lookup(W, self.input_x, name='character_embedding')
        #
        with tf.name_scope("cnn_pooling_stacks"):
            self.input_ = tf.cast(tf.expand_dims(self.input_, axis=-1), "float")
            for index, config in enumerate(conv_config):
                cnn_config = [self.input_] + config + [index]
                self.input_ = self.cnn_maxpool(*cnn_config)


        # flatten the result
        total_neurons = self.input_.get_shape()[1].value * self.input_.get_shape()[2].value
        self.input_ = tf.reshape(self.input_, shape=[-1, total_neurons])

        with tf.name_scope("fully_connected"):
            for index, num_out in enumerate(fc_config):
                self.input_ = self.fc_layers(self.input_, num_out, index)

        with tf.name_scope("scores_and_output"):
            width = self.input_.get_shape()[1].value
            stddev = 1/(width**1/2)
            w_out = tf.Variable(tf.random_uniform([width, num_classes], minval=-stddev, maxval=stddev),
                                name="output_weight")
            b_out = tf.Variable(tf.random_uniform(shape=[num_classes], minval=-stddev, maxval=stddev),
                                name="b")

            l2_loss += tf.nn.l2_loss(w_out)
            l2_loss += tf.nn.l2_loss(b_out)

            self.scores = tf.nn.xw_plus_b(self.input_, w_out, b_out, name='output_bias')
            self.softmax_scores = tf.nn.softmax(self.scores)
            self.output = tf.argmax(self.scores, axis=1)

        with tf.name_scope("loss_and_accuracy"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.label_y)
            self.loss = tf.reduce_mean(losses)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.output, tf.argmax(self.label_y, axis=1)), "float"))



    def cnn_maxpool(self, input_, kernel_size, num_filters, pooling_size, index):
        with tf.name_scope("cnn_maxpool_%s" % index):
            embedding_length = input_.get_shape()[2].value

            filter_size = [kernel_size, embedding_length, 1, num_filters]

            # W = tf.get_variable(name="cnn_weights_%s" % index,
            #                     initializer=tf.random_normal(shape=filter_size, mean=0, stddev=0.05))
            # b = tf.get_variable(name='cnn_bias_%s' % index,shape=[num_filters],
            #                    initializer=tf.random_normal_initializer(mean=0, stddev=0.05))


            # an alternative initializer
            stdv = 1 / ((kernel_size * embedding_length) ** 1/2)
            W = tf.Variable(tf.random_uniform(filter_size, minval=-stdv, maxval=stdv), dtype='float32',
                            name="cnn_weight_%s" % index)  # The kernel of the conv layer is a trainable vraiable
            b = tf.Variable(tf.random_uniform([num_filters], minval=-stdv, maxval=stdv),
                            name='cnn_b_%s' % index)

            # convolution
            conv = tf.nn.conv2d(input_, W, strides=[1, 1, 1, 1], padding='VALID', name='conv_%s' % index)


            # apply non-linear
            h = tf.nn.bias_add(conv, b)
            if pooling_size:
                pool = tf.nn.max_pool(h, ksize=[1, pooling_size, 1, 1], strides=[1, pooling_size, 1, 1], padding='VALID')
                return tf.transpose(pool, perm=[0, 1, 3, 2])
            else:
                return tf.transpose(h, perm=[0, 1, 3, 2])

    def fc_layers(self, input_, num_outputs, index):
        with tf.name_scope("fully_connected_layer_%s" % index):
            width = input_.get_shape()[1].value
            stddev = 1/(width**1/2)
            W = tf.get_variable(name="fully_connected_weight_1_%s" % index,
                                initializer=tf.random_uniform([width, num_outputs], minval=-stddev, maxval=stddev))
            b = tf.get_variable(name="fully_connected_bias_2_%s" % index,
                                initializer=tf.random_uniform([num_outputs], minval=-stddev, maxval=stddev))

            h = tf.nn.xw_plus_b(input_, W, b)

            return tf.nn.dropout(h, keep_prob=self.keep_prob)

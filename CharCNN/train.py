import tensorflow as tf
import datetime
import numpy as np
from CharCNN.CharCNN import CharCNN
from CharCNN.data_tool import data_tool
import os
import time


# Training procedures
class Training(data_tool, CharCNN):
    def __init__(self, data_path, truncated_length, conv_config, fc_config, batch_size, epoch_size, check_dir=None):
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        print('--- data_tool ---')
        data_tool.__init__(self, data_path=data_path, truncated_length=truncated_length)
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                print('--- model ---')
                CharCNN.__init__(self, sequence_length=truncated_length, char_vector=self.one_hot_vector,
                                 conv_config=conv_config, fc_config=fc_config,
                                 num_classes=self.data_y.shape[1])

                global_step = tf.Variable(0, name='global_step', trainable=False)
                # exponential decaying learning rate:
                # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
                lr = tf.train.exponential_decay(0.005, global_step=global_step, decay_steps=100, decay_rate=0.9)
                optimizer = tf.train.AdamOptimizer(lr)
                grads_and_vars = optimizer.compute_gradients(self.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step)

                # save/restore + temporary working directory
                saver = tf.train.Saver()
                if check_dir:
                    checkpoint_file = tf.train.latest_checkpoint(check_dir)
                    print('load existing checkpoint:', checkpoint_file)
                    saver.restore(sess, checkpoint_file)
                    temp_dir = os.path.split(check_dir)[-2]
                else:
                    temp_dir = str(int(time.time()))
                checkpoint_dir = os.path.join('CharCNN_runs', temp_dir, 'checkpoints')
                checkpoint_model_dir = checkpoint_dir + '/model.ckpt'

                # Summary for loss and accuracy
                loss_summary = tf.summary.scalar("loss", self.loss)
                acc_summary = tf.summary.scalar("accuracy", self.accuracy)

                # Train Summaries
                train_summary_op = tf.summary.merge([loss_summary, acc_summary])
                train_summary_dir = os.path.join("CharCNN_runs", temp_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                # Test Summaries
                test_summary_op = tf.summary.merge([loss_summary, acc_summary])
                test_summary_dir = os.path.join('CharCNN_runs', temp_dir, 'summaries', 'test')
                test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

                # define operations
                def train_(batch_x, batch_y):
                    feed_dict = {self.input_x: batch_x,
                                 self.label_y: batch_y,
                                 self.keep_prob: 0.5}

                    loss, _, accuracy, step, summaries = sess.run(
                        [self.loss, train_op, self.accuracy, global_step, train_summary_op],
                        feed_dict=feed_dict)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    train_summary_writer.add_summary(summaries, step)

                def test_():
                    feed_dict = {self.input_x: self.dev_x,
                                 self.label_y: self.dev_y,
                                 self.keep_prob: 1.0}

                    loss, accuracy, step, summaries = sess.run(
                        [self.loss, self.accuracy, global_step, test_summary_op],
                        feed_dict=feed_dict)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    test_summary_writer.add_summary(summaries, step)

                # initialize variable
                sess.run(tf.global_variables_initializer())

                # generate batches
                batches_all = self.generate_batches(data_x=self.train_x, data_y=self.train_y, epoch_size=self.epoch_size,
                                                    batch_size=self.batch_size, shuffle=True)
                total_amount = (len(self.train_x) // self.batch_size + 1) * self.epoch_size

                # generate test indices
                shuffle_indices = np.random.permutation(np.arange(len(self.test_x)))

                # training on batches
                print("Total step:", total_amount)
                for i, batch in enumerate(batches_all):
                    batch_x, batch_y = batch
                    train_(batch_x, batch_y)
                    current_step = tf.train.global_step(sess, global_step)
                    if i % 1000 == 0:
                        print('\nEvaluation:\n')
                        test_()
                        print("Writing model...\n")
                        saver.save(sess, checkpoint_model_dir, global_step=current_step)

                # evaluating the model on test data
                self.Evaluation_test(sess, window=500)

    def Evaluation_test(self, sess, window=500):
        # start testing and saving data
        data_size = len(self.test_x)
        result = []
        for i in range(data_size // window + 1):
            left_, right_ = i * window, min((i+1) * window, data_size)
            result.append(sess.run(self.output,
                                   feed_dict={self.input_x: self.test_x[left_:right_],
                                              self.keep_prob: 1.0}))
        result = np.concatenate(result, axis=0)
        print("Test data accuracy:", np.mean(np.equal(np.argmax(self.test_y, axis=1), result)))
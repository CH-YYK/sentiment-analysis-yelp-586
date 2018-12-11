import tensorflow as tf
from TextRNN.TextRNN import TextRNN
from TextRNN.data_utils import data_tool
import datetime
import time
import os
import numpy as np

class Training(data_tool, TextRNN):
    def __init__(self, train_data_path, test_data_path, corpus_path, word_vector_path, batch_size, epoch_size,
                 rnn_size, outdir='./',
                 Glove_path=None, check_dir=None):
        data_tool.__init__(self, train_data_path=train_data_path,
                           test_data_path=test_data_path, corpus_path=corpus_path, word_vector_path=word_vector_path, Glove_path=Glove_path)
        self.outdir = outdir
        with tf.Graph().as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                TextRNN.__init__(self, sequence_length=self.max_document_length,
                                 embedding_size=self.word_vec.shape[1],
                                 word_vector=self.word_vec, rnn_size=rnn_size)

                global_step = tf.Variable(0, name='global_step', trainable=False)
                lr = tf.train.exponential_decay(0.001, global_step=global_step, decay_steps=10000, decay_rate=1)
                optimizer = tf.train.AdamOptimizer(lr)
                grads_and_vars = optimizer.compute_gradients(self.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                # save/restore + temporary working directory
                saver = tf.train.Saver()
                if check_dir:
                    checkpoint_file = tf.train.latest_checkpoint(check_dir)
                    print('load existing checkpoint:', checkpoint_file)
                    saver.restore(self.sess, checkpoint_file)
                    temp_dir = os.path.split(check_dir)[-2]
                else:
                    temp_dir = str(int(time.time()))
                checkpoint_dir = os.path.join(outdir, 'TextCNN_runs', temp_dir, 'checkpoints')
                checkpoint_model_dir = checkpoint_dir + '/model.ckpt'

                # Summary for loss and accuracy
                loss_summary = tf.summary.scalar("loss", self.loss)
                acc_summary = tf.summary.scalar("accuracy", self.accuracy)

                # Train Summaries
                train_summary_op = tf.summary.merge([loss_summary, acc_summary])
                train_summary_dir = os.path.join(outdir, "TextRNN_runs", temp_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)

                # Test Summaries
                test_summary_op = tf.summary.merge([loss_summary, acc_summary])
                test_summary_dir = os.path.join(outdir, 'TextRNN_runs', temp_dir, 'summaries', 'test')
                test_summary_writer = tf.summary.FileWriter(test_summary_dir, self.sess.graph)

                # define operations
                def train_(batch_x, batch_y):
                    feed_dict = {self.input_x: batch_x,
                                 self.input_y: batch_y,
                                 self.real_length: self.real_words_length(batch_x),
                                 self.keep_prob: 0.5}

                    loss, _, accuracy, step, summaries = self.sess.run(
                        [self.loss, train_op, self.accuracy, global_step, train_summary_op],
                        feed_dict=feed_dict)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    train_summary_writer.add_summary(summaries, step)

                def test_():
                    feed_dict = {self.input_x: self.test_x[:500],
                                 self.input_y: self.test_y[:500],
                                 self.real_length: self.real_words_length(self.test_x[:500]),
                                 self.keep_prob: 1.0}

                    loss, accuracy, step, summaries = self.sess.run(
                        [self.loss, self.accuracy, global_step, test_summary_op],
                        feed_dict=feed_dict)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    test_summary_writer.add_summary(summaries, step)
                    return loss, accuracy

                # initialize
                self.sess.run(tf.global_variables_initializer())

                # generate batches
                batches_train = self.batches_generate(data_x=self.train_x, data_y=self.train_y, epoch_size=epoch_size,
                                                      batch_size=batch_size, shuffle=True)

                total = (len(self.train_y) // batch_size + 1) * epoch_size
                # training on batches
                print("Total step:", total)
                for i, batch in enumerate(batches_train):
                    batch_x, batch_y = batch
                    train_(batch_x, batch_y)
                    current_step = tf.train.global_step(self.sess, global_step)
                    if i % 100 == 0 and i > 0:
                        print('\nEvaluation:\n')
                        loss, accuracy = test_()
                        # print("Writing model...\n")
                        # saver.save(self.sess, checkpoint_model_dir, global_step=current_step)
                self.Evaluation_test(self.sess, window=500, save=temp_dir)

    def real_words_length(self, batches):
        return np.ceil([np.argmin(batch.tolist() + [0]) for batch in batches.reshape((-1, batches.shape[-1]))])

    def Evaluation_test(self, sess, window=500, save=None):
        # start testing and saving data
        data_size = len(self.test_x)
        result = []
        for i in range(data_size // window + 1):
            left_, right_ = i * window, min((i+1) * window, data_size)
            result.append(sess.run(self.output,
                                   feed_dict={self.input_x: self.test_x[left_:right_],
                                              self.keep_prob: 1.0,
                                              self.real_length: self.real_words_length(self.test_x[left_:right_])}))
        result = np.concatenate(result, axis=0)
        print("Test data accuracy:", np.mean(np.equal(np.argmax(self.test_y, axis=1), result)))
        self.test['pred'] = result+1
        self.test.to_csv(os.path.join(self.outdir, 'TextRNN_runs', save, 'textrnn.tsv'), sep='\t')

if __name__ == '__main__':
    train_data_path = "../data/business_reviews2017_train.tsv"
    test_data_path = "../data/business_reviews2017_test.tsv"
    corpus_path = "corpus.pkl"
    word_vect = "word_vector.npy"
    test = Training(train_data_path, test_data_path, corpus_path, word_vect, epoch_size=20)

import tensorflow as tf
from .category_TextCNN import TextCNN
from .data_utils import data_tool
import datetime, time
import os
import numpy as np


class Training(data_tool, TextCNN):
    def __init__(self, train_data_path, test_data_path, corpus_path, word_vector_path,
                 category_corpus_path, category_vector_path,
                 batch_size, epoch_size, filters_sizes=[3, 4, 5], num_filters=256, check_dir=None, outdir='./'):
        data_tool.__init__(self, train_data_path=train_data_path, test_data_path=test_data_path,
                           corpus_path=corpus_path, word_vector_path=word_vector_path,
                           category_corpus_path=category_corpus_path, category_vector=category_vector_path)
        self.outdir = outdir
        with tf.Graph().as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                TextCNN.__init__(self, sequence_length=self.max_document_length,
                                 embedding_size=self.word_vec.shape[1],
                                 word_vector=self.word_vec,
                                 filter_sizes=filters_sizes,
                                 num_filters=num_filters,
                                 category_length=self.max_category_length,
                                 category_vector=self.category_vector)

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
                checkpoint_dir = os.path.join(outdir, 'TextCNN__cat_runs', temp_dir, 'checkpoints')
                checkpoint_model_dir = checkpoint_dir + '/model.ckpt'

                # Summary for loss and accuracy
                loss_summary = tf.summary.scalar("loss", self.loss)
                acc_summary = tf.summary.scalar("accuracy", self.accuracy)

                # Train Summaries
                train_summary_op = tf.summary.merge([loss_summary, acc_summary])
                train_summary_dir = os.path.join(outdir, "TextCNN_cat_runs", temp_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)

                # Test Summaries
                test_summary_op = tf.summary.merge([loss_summary, acc_summary])
                test_summary_dir = os.path.join(outdir, 'TextCNN_cat_runs', temp_dir, 'summaries', 'test')
                test_summary_writer = tf.summary.FileWriter(test_summary_dir, self.sess.graph)

                # define operations
                def train_(batch_x, batch_y, batch_category, total):
                    feed_dict = {self.input_x: batch_x,
                                 self.input_y: batch_y,
                                 self.input_category: batch_category,
                                 self.keep_prob: 0.5}

                    loss, _, accuracy, step, summaries = self.sess.run(
                        [self.loss, train_op, self.accuracy, global_step, train_summary_op],
                        feed_dict=feed_dict)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}/{}, loss {:g}, acc {:g}".format(time_str, step, total, loss, accuracy))
                    train_summary_writer.add_summary(summaries, step)

                def test_():
                    feed_dict = {self.input_x: self.test_x[:500],
                                 self.input_y: self.test_y[:500],
                                 self.input_category: self.category_test[:500],
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
                batches_train = self.batches_generate(data_x=self.train_x, data_y=self.train_y,
                                                      categories=self.category_train, epoch_size=epoch_size,
                                                      batch_size=batch_size, shuffle=True)

                total = (len(self.train_y) // batch_size + 1) * epoch_size
                # training on batches
                print("Total step:", total)
                for i, batch in enumerate(batches_train):
                    batch_x, batch_y, batch_category = batch
                    train_(batch_x, batch_y, batch_category, total)
                    current_step = tf.train.global_step(self.sess, global_step)
                    if i % 100 == 0 and i > 0:
                        print('\nEvaluation:\n')
                        loss, accuracy = test_()
                        # print("Writing model...\n")
                        # saver.save(self.sess, checkpoint_model_dir, global_step=current_step)
                    if current_step == 3100:
                        break
                self.Evaluation_test(self.sess, window=500, save=temp_dir)

    def Evaluation_test(self, sess, window=500, save=''):
        # start testing and saving data
        data_size = len(self.test_x)
        result = []
        for i in range(data_size // window + 1):
            left_, right_ = i * window, min((i+1) * window, data_size)
            result.append(sess.run(self.output,
                                   feed_dict={self.input_x: self.test_x[left_:right_],
                                              self.input_category: self.category_test[left_:right_],
                                              self.keep_prob: 1.0}))
        result = np.concatenate(result, axis=0)
        print("Test data accuracy:", np.mean(np.equal(np.argmax(self.test_y, axis=1), result)))
        self.test['pred'] = result + 1
        self.test.to_csv(os.path.join(self.outdir, 'TextCNN_cat_runs', save, "textcnn_cat.tsv"), sep='\t')

if __name__ == '__main__':
    train_data_path = "../data/business_reviews2017_trunc_train.tsv"
    test_data_path = "../data/business_reviews2017_trunc_test.tsv"
    corpus_path = "../data/corpus.pkl"
    word_vect = "../data/word_vector.npy"
    category_corpus_path = "../data/category_corpus.pkl"
    category_vector_path = "../data/category_vector.npy"
    test = Training(train_data_path, test_data_path, corpus_path, word_vect, category_corpus_path, category_vector_path,
                    batch_size=64, epoch_size=32)

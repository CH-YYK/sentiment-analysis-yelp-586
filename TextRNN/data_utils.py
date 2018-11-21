import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn
import pickle, os
import pandas as pd

class data_tool(object):

    def __init__(self, train_data_path, test_data_path, corpus_path, word_vector_path, Glove_path=None):
        # load data (could changed)
        self.train = pd.read_table(train_data_path, sep='\t')[['text', 'class']]
        self.test = pd.read_table(test_data_path, sep='\t')[['text', 'class']]

        # load corpus
        if os.path.isfile(corpus_path):
            self.corpus = pickle.load(open(corpus_path, 'rb'))
            self.word_vec = np.load(word_vector_path)
        else:
            self.corpus, self.word_vec = self.extractGlove(Glove_path)

        # add unknown tokens, and 0 initialization
        self.corpus['<UNK>'] = 0
        self.word_vec = np.concatenate([np.ones((1, self.word_vec.shape[1])), self.word_vec], axis=0)

        # tokenize texts
        self.max_document_length = max([len(text.split()) for text in self.train['text']] +
                                       [len(text.split()) for text in self.test['text']])
        vocab_preprocessor = learn.preprocessing.VocabularyProcessor(max_document_length=self.max_document_length,
                                                                     tokenizer_fn=self.tokenize,
                                                                     vocabulary=self.corpus)
        self.train_x = np.array(list(vocab_preprocessor.transform(self.train['text'])))
        self.test_x = np.array(list(vocab_preprocessor.transform(self.test['text'])))

        label_encoder = LabelBinarizer().fit([1, 2, 3, 4, 5])
        self.train_y = label_encoder.transform(self.train['class'])
        self.test_y = label_encoder.transform(self.test['class'])

    def extractGlove(self, glove_path):
        corpus, vectors = [], []
        glove = open(glove_path, 'r').readlines()
        for line in glove:
            seq = line.split()
            corpus.append(seq[0])
            vectors.append([float(num) for num in seq[1:]])

        # assign indices to words, make word_vector matrix
        corpus = {word: i+1 for i, word in enumerate(corpus)}
        vectors = np.array(vectors)
        return corpus, vectors

    def tokenize(self, iterator):
        for i in iterator:
            lis = []
            for j in i.split(' '):
                if j not in self.corpus:
                    j = "<UNK>"
                lis.append(j)
            yield lis

    def batches_generate(self, data_x, data_y, epoch_size=10, batch_size=64, shuffle=True):
        """
        generate training batches
        """
        data_size = len(data_y)
        num_batches = data_size // batch_size + 1

        for i in range(epoch_size):
            if shuffle:
                np.random.seed(1000)
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffle_data_x, shuffle_data_y = data_x[shuffle_indices], data_y[shuffle_indices]
            else:
                shuffle_data_x, shuffle_data_y = data_x, data_y

            for j in range(num_batches):
                start_index = j * batch_size
                end_index = min((j + 1) * batch_size, data_size)
                batch_x = shuffle_data_x[start_index: end_index]
                batch_y = shuffle_data_y[start_index: end_index]
                yield batch_x, batch_y


if __name__ == '__main__':
    train_data_path = "../data/business_reviews2017_train.tsv"
    test_data_path = "../"
    corpus_path = "corpus.pkl"
    word_vect = "word_vector.npy"

    test = data_tool(train_data_path=train_data_path, test_data_path=test_data_path, corpus_path=corpus_path,
                     word_vector_path=word_vect)
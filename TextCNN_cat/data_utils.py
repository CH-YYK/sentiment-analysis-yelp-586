import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn
import pickle, os
import pandas as pd

class data_tool(object):

    def __init__(self, train_data_path, test_data_path, corpus_path, word_vector_path,
                 category_corpus_path, category_vector):
        # load data (could changed)
        self.train = pd.read_table(train_data_path, sep='\t')[['text', 'class', 'categories']]
        self.test = pd.read_table(test_data_path, sep='\t')[['text', 'class', 'categories']]

        # ---------------------- prepare for texts ------------------------
        # load corpus for texts
        self.corpus = pickle.load(open(corpus_path, 'rb'))
        self.word_vec = np.load(word_vector_path)

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

        # ---------------------- prepare for categories ------------------------
        # load corpus for categories
        self.category_corpus = pickle.load(open(category_corpus_path, 'rb'))
        self.category_vector = np.load(category_vector)
        self.category_corpus['<UNK>'] = 0

        # tokenize categories
        self.max_category_length = max(self.train['categories'].apply(lambda x: len(x.split(', '))).max(),
                                       self.test['categories'].apply(lambda x: len(x.split(', '))).max())
        category_processor = learn.preprocessing.VocabularyProcessor(max_document_length=self.max_category_length,
                                                                     tokenizer_fn=self.tokenize_category,
                                                                     vocabulary=self.category_corpus)
        self.category_train = np.array(list(category_processor.transform(self.train['categories'])))
        self.category_test = np.array(list(category_processor.transform(self.test['categories'])))


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

    def tokenize_category(self, iterator):
        for i in iterator:
            lis = []
            for j in i.split(', '):
                if j not in self.category_corpus:
                    j = "<UNK>"
                lis.append(j)
            yield lis

    def batches_generate(self, data_x, data_y, categories, epoch_size=10, batch_size=64, shuffle=True):
        """
        generate training batches
        """
        data_size = len(data_y)
        num_batches = data_size // batch_size + 1

        for i in range(epoch_size):
            if shuffle:
                np.random.seed(1000)
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffle_data_x, shuffle_data_y, shuffle_category = \
                    data_x[shuffle_indices], data_y[shuffle_indices], categories[shuffle_indices]
            else:
                shuffle_data_x, shuffle_data_y, shuffle_category = data_x, data_y, categories

            for j in range(num_batches):
                start_index = j * batch_size
                end_index = min((j + 1) * batch_size, data_size)
                batch_x = shuffle_data_x[start_index: end_index]
                batch_y = shuffle_data_y[start_index: end_index]
                batch_category = shuffle_category[start_index: end_index]
                yield batch_x, batch_y, batch_category


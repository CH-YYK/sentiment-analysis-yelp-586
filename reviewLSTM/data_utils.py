import numpy as np
from tensorflow.contrib import learn
import pandas as pd
import pickle, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

class data_tool(object):

    def __init__(self, train_data_path, test_data_path, corpus_path, word_vector_path, Glove_path, review_size, len_words):
        self.review_size = review_size
        self.len_words = len_words

        # load data and split
        # self.data = pd.read_table(data_path, sep='\t')[['text', 'class']]
        # self.train, self.test = train_test_split(self.data)
        self.train = pd.read_table(train_data_path, sep='\t')[['text', 'class']]
        self.test = pd.read_table(test_data_path, sep='\t')[['text', 'class']]

        # load corpus
        if os.path.isfile(corpus_path):
            self.corpus = pickle.load(open(corpus_path, 'rb'))
            self.word_vec = np.load(word_vector_path)
        else:
            self.corpus, self.word_vec = self.extractGlove(Glove_path)
        self.corpus['<UNK>'] = 0
        self.word_vec = np.concatenate([np.zeros((1, self.word_vec.shape[1])), self.word_vec], axis=0)

        # build vocabulary processor
        self.vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=len_words,
                                                                       vocabulary=self.corpus,
                                                                       tokenizer_fn=self.tokenize)
        self.train_x = self.split_reviews(self.train['text'])
        self.test_x = self.split_reviews(self.test['text'])

        label_binorizer = LabelBinarizer().fit([1, 2, 3, 4, 5])
        self.train_y = label_binorizer.transform(self.train['class'])
        self.test_y = label_binorizer.transform(self.test['class'])

    def split_reviews(self, texts, separator='\.'):
        data = []
        for text in texts:
            reviews_ = np.array(list(self.vocab_processor.transform(text.split(separator))))

            reviews_ = np.concatenate([reviews_, np.zeros((self.review_size-len(reviews_), self.len_words))], axis=0)
            data.append(reviews_)
        return np.array(data)

    def extractGlove(self, glove_path):
        corpus, vectors = [], []
        glove = open(glove_path, 'r').readlines()
        for line in glove:
            seq = line.split()
            corpus.append(seq[0])
            vectors.append([float(num) for num in seq[1:]])

        # assign indices to words, make word_vector matrix
        corpus = {word: i + 1 for i, word in enumerate(corpus)}
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
    data_path = "../data/business_reviews2017.tsv"
    corpus_path = "../data/corpus.pkl"
    word_vect = "../data/word_vector.npy"

    test = data_tool(data_path=data_path, corpus_path=corpus_path, word_vector_path=word_vect, review_size=5, len_words=100, Glove_path=None)

import pandas as pd
import re
import numpy as np
import os


class data_tool(object):

    def __init__(self, data_path, truncated_length, train_set_ratio=0.8, dev_set_size=600):
        # load data and corpus
        self.char_dict = self.character_corpus()

        # assign label and texts
        if os.path.isfile('dataset_x.npy') and os.path.isfile('dataset_y.npy'):
            print('Load existing index set')
            self.data_x = np.load('dataset_x.npy')
            self.data_y = np.load('dataset_y.npy')
        else:
            self.data = self.clean_data(data_path)
            print('Converting text to index')
            self.data_x = np.array([self.text2index(sent[:truncated_length], self.char_dict, truncated_length)
                                    for sent in self.data.iloc[:, 0]])
            label = [int(label) for label in self.data.iloc[:, 1]]

            polarity = lambda x: int(x >= 3)
            self.data_y = [[0] * 1 for _ in label]
            _ = [self.data_y[i].insert(polarity(j), 1) for i, j in enumerate(label)]
            self.data_y = np.array(self.data_y)
            print('Save index set')
            np.save('dataset_x', self.data_x)
            np.save('dataset_y', self.data_y)

        # split raw to train/dev+test
        print("Splitting data set into train/dev/test")
        train_split = int(self.data_x.shape[0] * train_set_ratio)
        self.train_x, self.train_y = self.data_x[:train_split], self.data_y[:train_split]

        # split dev/test
        self.dev_x, self.dev_y = self.data_x[train_split:], self.data_y[train_split:]
        self.test_x, self.test_y = self.dev_x[dev_set_size:], self.dev_y[dev_set_size:]
        self.dev_x, self.dev_y = self.dev_x[:dev_set_size], self.dev_y[:dev_set_size]

        print("train: {}, dev: {}, test: {}".format(self.train_x.shape, self.dev_x.shape, self.test_x.shape))
        # one-hot-vector
        self.one_hot_vector = self.to_one_hot(self.char_dict, truncated_length)

    def clean_data(self, data_path):
        """
        load dataset from data_path and then extract the target columns (used when loading data)
        :param data_path: path to data file
        :return: datatable with target columns
        """
        print('Load dataset...')
        data = pd.read_table(data_path, sep='\t', nrows=300000)
        return data[['text', 'stars']].dropna(axis=0)

    def character_corpus(self):
        """
        build character-corpus
        """
        char_dict = {char: index + 1 for index, char
                     in enumerate("abcdefghijklmnopqrstuvwxyz0123456789,;.!?:\'\"/\\|_@#$%^&*~`+-=<>()[]{}")}
        char_dict['\n'] = 69
        return char_dict

    def text2index(self, text, vocab_dict, truncated_length):
        """
        tokenization: tokenize sentence to character indices
        """
        text = self.clean_str(text)
        tmp = [0] * truncated_length
        for i in range(1, len(text) + 1):
            tmp[i - 1] = vocab_dict.get(text[-i].lower(), 0)
        return tmp

    def clean_str(self, string):
        """
        clean strings. ex. add newline characters (used in tokenization)
        """
        return re.sub("[a-zA-Z]\\+[a-zA-Z]", '\n', string)

    def to_one_hot(self, char_dict, truncated_length):
        """
        build one-hot vector embedding for the character-corpus
        """
        tmp = np.zeros([truncated_length, char_dict.keys().__len__()])
        for i, j in enumerate(char_dict.values()):
            tmp[i, j - 1] = 1
        return np.concatenate([np.zeros([truncated_length, 1]), tmp], axis=1)

    def save_data(self, result):
        real_value = np.argmax(self.test_y, axis=1).tolist()
        predicted = [i for i in result.reshape(-1).tolist()]
        return pd.DataFrame(list(zip(real_value, predicted)), columns=['real_values', 'predicted_values'])

    def generate_batches(self, data_x, data_y, epoch_size, batch_size, shuffle=False):
        """
        generate training batches
        """
        data_size = len(data_x)
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
    data_path = '/Users/kayleyang/Desktop/sentiment-analysis-yelp-586/data/yelp_academic_dataset_review.tsv'
    test = data_tool(data_path, 1014)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import argparse
import numpy as np
import pandas as pd

## $ python analysis/create_train_test.py -d data/business_reviews2017_trunc.tsv -o data/business_reviews2017_trunc


class overSampling(object):

    def __init__(self, data_path, out_path, verbose=True):
        self.data = pd.read_table(data_path, sep='\t', usecols=['class', 'text', 'categories'])
        np.random.seed(1000)

        self.train, self.test = train_test_split(self.data, train_size=0.7)

        # oversampling train_set
        ros = RandomOverSampler(random_state=0)
        self.labels = self.train['class']
        self.train_sampled, self.train_labels_resampled = ros.fit_resample(self.train, self.labels)
        self.train = pd.DataFrame(self.train_sampled, columns=list(self.train))

        #
        if verbose:
            self.train.to_csv(out_path+'_train.tsv', sep='\t', index=False)
            self.test.to_csv(out_path+'_test.tsv', sep='\t', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, help="the path to dataset")
    parser.add_argument("-o", "--output", type=str, help="the dir to write data")
    args = vars(parser.parse_args())

    tmp = overSampling(args['data'], args['output'], verbose=True)

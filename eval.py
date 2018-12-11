import argparse
from sklearn import metrics
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--true', default="data/business_reviews2017_trunc_test.tsv", help="The file which has true classes")
parser.add_argument('--pred', required=True, help="The file which has predicted classes")
args = vars(parser.parse_args())

true = pd.read_csv(args['true'], sep='\t')['class']
pred = pd.read_csv(args['pred'], sep='\t')['pred']

def output_result(true, pred):
    print(metrics.classification_report(true, pred, digits=10))
    print(metrics.f1_score(true, pred, average='micro'))

if __name__ == '__main__':
    output_result(true, pred)
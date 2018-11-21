import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=64, help='Set the batch size during training')
parser.add_argument('-e', '--epoch_size', type=int, default=30, help='Set the epoch size during training')
parser.add_argument('-c', '--check_dir', default=None, help="The path where the model will be restored")
parser.add_argument('-m', '--Model', default='CharCNN', help='The name of the model to be evaluated')
parser.add_argument('-d', '--Data', default='./data/business_reviews2017.tsv', help='The path to the data set')
parser.add_argument('--train', default='./data/business_reviews2017_train.tsv', help='The path to training dataset')
parser.add_argument('--test', default='./data/business_reviews2017_test.tsv', help="The path to testing dataset")
parser.add_argument('--corpus', default="data/corpus.pkl", help='path to corpus')
parser.add_argument('--outdir', default='../TextData/', help='The directory to store summaries')

args = vars(parser.parse_args())


class global_conf:
    """
    define global configurations
    """
    batch_size = args['batch_size']
    epoch_size = args['epoch_size']
    check_dir = args['check_dir']
    Model_name = args['Model']
    outDir = args['outdir']

if __name__ == '__main__':
    if args['Model'] == 'CharCNN':
        from CharCNN import train, conf
        conf.data_path = args['Data']
        train.Training(data_path=conf.data_path, truncated_length=conf.truncated_length, conv_config=conf.conv_config,
                       fc_config=conf.fc_config,
                       batch_size=global_conf.batch_size, epoch_size=global_conf.epoch_size, check_dir=global_conf.check_dir)
        del train, conf
    elif args['Model'].lower() == 'textcnn':
        from TextCNN import train, conf
        conf.train_data_path = args['train']
        conf.test_data_path = args['test']
        conf.corpus_path = args['corpus']
        conf.batch_size = global_conf.batch_size
        conf.epoch_size = global_conf.epoch_size
        conf.outDir = global_conf.outDir

        train.Training(train_data_path=conf.train_data_path, test_data_path=conf.test_data_path,
                       corpus_path=conf.corpus_path, word_vector_path=conf.word_vect,
                       batch_size=conf.batch_size, epoch_size=conf.epoch_size, outdir=conf.outDir,
                       filters_sizes=conf.num_filters, num_filters=conf.num_filters)
    elif args['Model'].lower() == 'textrnn':
        from TextRNN import train, conf
        conf.train_data_path = args['train']
        conf.test_data_path = args['test']
        conf.corpus_path = args['corpus']
        conf.batch_size = global_conf.batch_size
        conf.epoch_size = global_conf.epoch_size
        conf.outDir = global_conf.outDir

        train.Training(train_data_path=conf.train_data_path, test_data_path=conf.test_data_path,
                       corpus_path=conf.corpus_path, word_vector_path=conf.word_vect,
                       rnn_size=conf.rnn_size,
                       batch_size=conf.batch_size, epoch_size=conf.epoch_size, outdir=conf.outDir)
    elif args['Model'].lower() == "reviewlstm":
        from reviewLSTM import train, conf
        conf.train_data_path = args['train']
        conf.test_data_path = args['test']
        conf.corpus_path = args['corpus']
        conf.batch_size = global_conf.batch_size
        conf.epoch_size = global_conf.epoch_size
        conf.outDir = global_conf.outDir
        train.Training(train_data_path=conf.train_data_path, test_data_path=conf.test_data_path,
                       corpus_path=conf.corpus_path, word_vector_path=conf.word_vect,
                       rnn_size=conf.rnn_size,
                       batch_size=conf.batch_size, epoch_size=conf.epoch_size, outdir=conf.outDir)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=256, help='Set the batch size during training')
parser.add_argument('-e', '--epoch_size', type=int, default=5, help='Set the epoch size during training')
parser.add_argument('-c', '--check_dir', default=None, help="The path where the model will be restored")
parser.add_argument('-m', '--Model', default='CharCNN', help='The name of the model to be evaluated')
parser.add_argument('-d', '--Data', default='./data/reviews_2015.tsv', help='The path to the data set')

args = vars(parser.parse_args())


class global_conf:
    """
    define global configurations
    """
    batch_size = args['batch_size']
    epoch_size = args['epoch_size']
    check_dir = args['check_dir']
    Model_name = args['Model']

if __name__ == '__main__':
    if args['Model'] == 'CharCNN':
        from CharCNN import train, conf
        conf.data_path = args['Data']
        train.Training(data_path=conf.data_path, truncated_length=conf.truncated_length, conv_config=conf.conv_config,
                       fc_config=conf.fc_config,
                       batch_size=global_conf.batch_size, epoch_size=global_conf.epoch_size, check_dir=global_conf.check_dir)
        del train, conf


print('Load CharCNN models')

class conf:
    data_path = 'reviews_2015.tsv'
    # Model configuration
    truncated_length = 1014
    conv_config = [[7, 256, 3],
                   [7, 256, 3],
                   [3, 256, None],
                   [3, 256, None],
                   [3, 256, None],
                   [3, 256, 3]]
    fc_config = [1024, 1024]


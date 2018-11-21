class conf:
    batch_size = 64
    epoch_size = 30
    filter_sizes = [3, 4, 5]
    num_filters = 128

    train_data_path = 'data/business_reviews2017_train.tsv'
    test_data_path = 'data/business_reviews2017_test.tsv'
    corpus_path = 'data/corpus.pkl'
    word_vect = 'data/word_vector.npy'
    outDir = './'


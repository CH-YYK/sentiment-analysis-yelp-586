class conf:
    batch_size = 64
    epoch_size = 30
    rnn_size = 128

    #
    len_words = 100

    train_data_path = 'data/business_reviews2017_train.tsv'
    test_data_path = 'data/business_reviews2017_test.tsv'
    corpus_path = 'data/corpus.pkl'
    word_vect = 'data/word_vec.npy'
    outDir = './'

    category_corpus_path = './data/category_corpus.pkl'
    category_vector_path = './data/category_vector.npy'

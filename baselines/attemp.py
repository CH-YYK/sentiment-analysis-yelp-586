from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas as pd
import numpy as np
import string

# load dataset
with open('corpus.txt') as f:
    labels, texts = [], []
    for i, line in enumerate(f.readlines()):
        content = line.split()
        labels.append(content[0])
        texts.append(' '.join(content[1:]))

# create dataframe
trainDF = pd.DataFrame(list(zip(labels, texts)), columns=['label', 'text'])


# split data into train/test
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

# encode y to categories
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

## features

# 1. use word counts as features counter
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])

# use word counter to transform train_set and valid set
xtrain_count = count_vect.transform(train_x)
xvalid_count = count_vect.transform(valid_x)

# 2. use TF-IDF as feature sets

# word-level TF-IDF
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf = tfidf_vect.transform(train_x)
xvalid_tfidf = tfidf_vect.transform(valid_x)

# ngram-level TF-IDF
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                                   ngram_range=(2, 3), max_features=5000)
tfidf_vect_ngram = tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram = tfidf_vect_ngram.transform(valid_x)

# char-level TF-IDF
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(train_x)
xvalid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(valid_x)


## Model
def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y, is_neural_net=False):
    """
    fit the training dataset on the classifier
    """
    classifier.fit(feature_vector_train, label)

    # predictions
    predictions = classifier.predict(feature_vector_valid)

    #
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    return metrics.accuracy_score(predictions, valid_y)

# ----------------------- Naive Bayes ---------------------------

## Naive Bayes - on word count vector
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count, valid_y)
print("NB - WordCount:", accuracy)

## Naive Bayes - on word-level TF-IDF
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
print("NB - WordTF-IDF:", accuracy)

## Naive Bayes - on ngram-level TF-IDF
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y,
                       xvalid_tfidf_ngram, valid_y)
print("NB - NgramTF-IDF:", accuracy)

## Naive Bayes - on ngram-char-level TF-IDF
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y,
                       xvalid_tfidf_ngram_chars, valid_y)
print("NB - NgramCharTF-IDF:", accuracy)

# ----------------------- LogisticRegression ---------------------------

## LR - on word count vector
accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count, valid_y)
print("LR - WordCount:", accuracy)

## LR - on word-level TF-IDF
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
print("LR - WordTF-IDF:", accuracy)

## LR - on ngram-level TF-IDF
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y,
                       xvalid_tfidf_ngram, valid_y)
print("LR - NgramTF-IDF:", accuracy)

## LR - on ngram-char-level TF-IDF
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y,
                       xvalid_tfidf_ngram_chars, valid_y)
print("LR - NgramCharTF-IDF:", accuracy)

# ----------------------- SVM ---------------------------

## SVM - on word count vector
accuracy = train_model(svm.SVC(), xtrain_count, train_y, xvalid_count, valid_y)
print("SVM - WordCount:", accuracy)

## SVM - on word-level TF-IDF
accuracy = train_model(svm.SVC(), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
print("SVM - WordTF-IDF:", accuracy)

## SVM - on ngram-level TF-IDF
accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y,
                       xvalid_tfidf_ngram, valid_y)
print("SVM - NgramTF-IDF:", accuracy)

## SVM - on ngram-char-level TF-IDF
accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram_chars, train_y,
                       xvalid_tfidf_ngram_chars, valid_y)
print("SVM - NgramCharTF-IDF:", accuracy)
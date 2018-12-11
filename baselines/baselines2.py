# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 13:13:51 2018

@author: jacky
"""
# set working directory
#import os 
#os.chdir("C:\\Users\\jacky\\Desktop")

# import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model, metrics, svm
#from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import GaussianNB
import nltk
from nltk.corpus import stopwords
#import pickle

# load into dataframe
mydf_train = pd.read_csv(r"data/business_reviews2017_train.tsv", delimiter = "\t", encoding = "utf-8")
mydf_test = pd.read_csv(r"data/business_reviews2017_test.tsv", delimiter = "\t", encoding = "utf-8")
mydf = pd.concat([mydf_train, mydf_test]) #combine both dfs to get the whole dataset
print(mydf_train.shape)
print(mydf_test.shape)
print(mydf.shape)

# shuffle rows
shuffle_index = np.random.permutation(range(mydf.shape[0]))
mydf = mydf.iloc[shuffle_index]

# label encoding
#mydf['class'] = preprocessing.label_binarize(mydf['class'] >= 4, [0,1])

# train/valid split (80:20 split)
#train_x, valid_x, train_y, valid_y = model_selection.train_test_split(mydf['text'], mydf['class'], test_size = 0.2, random_state = 12345)
train_x = mydf_train['text']
train_y = mydf_train['class']
valid_x = mydf_test['text']
valid_y = mydf_test['class'] 

print('BOW')
# 1: count of words feature set (BOW)
count_vect = CountVectorizer(analyzer='word', token_pattern = r'\w{1,}', stop_words = stopwords.words('english'), max_features = 200)
_ = count_vect.fit(mydf['text'])
xtrain_count = count_vect.transform(train_x)
xvalid_count = count_vect.transform(valid_x)

print('TFIDF')
# 2: TFIDF
# word-level TF-IDF (BOW-TFIDF)
tfidf_vect = TfidfVectorizer(analyzer = 'word', token_pattern = r'\w{1,}', max_features = 200, stop_words = stopwords.words('english'))
tfidf_vect.fit(mydf['text'])
xtrain_tfidf = tfidf_vect.transform(train_x)
xvalid_tfidf = tfidf_vect.transform(valid_x)

print('Bigrams-TFIDF')
# ngram-level TF-IDF (Bigrams-TFIDF)
tfidf_vect_ngram = TfidfVectorizer(analyzer = 'word', token_pattern = r'\w{1,}', ngram_range = (2, 3), max_features = 400, stop_words = stopwords.words('english'))
tfidf_vect_ngram = tfidf_vect_ngram.fit(mydf['text'])
xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram = tfidf_vect_ngram.transform(valid_x)

#print('Character-TFIDF')
# char-level TF-IDF (Character-TFIDF)
#tfidf_vect_ngram_chars = TfidfVectorizer(analyzer = 'char', token_pattern = r'\w{1,}', ngram_range = (2, 3), max_features = 500, stop_words = stopwords.words('english'))
#tfidf_vect_ngram_chars.fit(mydf['text'])
#xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(train_x)
#xvalid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(valid_x)

# baseline models
def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y, is_neural_net = False):
    """
    fit the training dataset on the classifier
    """
    classifier.fit(feature_vector_train, label)
    #prediction
    predictions = classifier.predict(feature_vector_valid)
    #space for ANNs
    if is_neural_net:
        predictions = predictions.argmax(axis = -1)
    
    return metrics.accuracy_score(predictions, valid_y), metrics.classification_report(valid_y, predictions, digits=4), \
           metrics.precision_score(valid_y, predictions, average = 'micro'), metrics.f1_score(valid_y, predictions, average = 'micro')

# Naive Bayes
print('Naive Bayes')
accuracy_NB_count = train_model(GaussianNB(), xtrain_count.toarray(), train_y, xvalid_count.toarray(), valid_y)
print(accuracy_NB_count)
accuracy_NB_TFIDF = train_model(GaussianNB(), xtrain_tfidf.toarray(), train_y, xvalid_tfidf.toarray(), valid_y)
print(accuracy_NB_TFIDF)
accuracy_NB_ngram_TFIDF = train_model(GaussianNB(), xtrain_tfidf_ngram.toarray(), train_y, xvalid_tfidf_ngram.toarray(), valid_y)
print(accuracy_NB_ngram_TFIDF)
#accuracy_NB_char_TFIDF = train_model(GaussianNB(), xtrain_tfidf_ngram_chars.toarray(), train_y, xvalid_tfidf_ngram_chars.toarray(), valid_y)
#print(accuracy_NB_char_TFIDF)

# Logistic Regression
print('Logistic Regression')
accuracy_LR_count = train_model(linear_model.LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs'), xtrain_count, train_y, xvalid_count, valid_y)
print(accuracy_LR_count)
accuracy_LR_tfidf = train_model(linear_model.LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs'), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
print(accuracy_LR_tfidf)
accuracy_LR_ngram_TFIDF = train_model(linear_model.LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs'), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, valid_y)
print(accuracy_LR_ngram_TFIDF)
#accuracy_LR_char_TFIDF = train_model(linear_model.LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs'), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars, valid_y)
#print(accuracy_LR_char_TFIDF)

xvalid_count_ind = (xvalid_count > 5).toarray().astype('float')
xtrain_count_ind = (xtrain_count > 5).toarray().astype('float')
# SVM with rbf kernel
print('SVM with rbf kernel')
accuracy_svm_count = train_model(svm.SVC(max_iter=10000, tol=0.001), xtrain_count, train_y, xvalid_count, valid_y)
print(accuracy_svm_count)
accuracy_svm_tfidf = train_model(svm.SVC(), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
print(accuracy_svm_tfidf)
accuracy_svm_ngram_TFIDF = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, valid_y)
print(accuracy_svm_ngram_TFIDF)
#accuracy_svm_char_TFIDF = train_model(svm.SVC(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars, valid_y)
#print(accuracy_svm_char_TFIDF)

# SVM with poly kernel
print('SVM with poly kernel')
accuracy_svm_count_2 = train_model(svm.SVC(kernel = 'poly'), xtrain_count, train_y, xvalid_count, valid_y)
print(accuracy_svm_count_2)
accuracy_svm_tfidf_2 = train_model(svm.SVC(kernel = 'poly'), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
print(accuracy_svm_tfidf_2)
accuracy_svm_ngram_TFIDF_2 = train_model(svm.SVC(kernel = 'poly'), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, valid_y)
print(accuracy_svm_ngram_TFIDF_2)
#accuracy_svm_char_TFIDF_2 = train_model(svm.SVC(kernel = 'poly'), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars, valid_y)
#print(accuracy_svm_char_TFIDF_2)

# SVM with linear kernel
print('SVM with linear kernel')
accuracy_svm_count_1 = train_model(svm.SVC(kernel = 'linear'), xtrain_count, train_y, xvalid_count, valid_y)
print(accuracy_svm_count_1)
accuracy_svm_tfidf_1 = train_model(svm.SVC(kernel = 'linear'), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
print(accuracy_svm_tfidf_1)
accuracy_svm_ngram_TFIDF_1 = train_model(svm.SVC(kernel = 'linear'), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, valid_y)
print(accuracy_svm_ngram_TFIDF_1)
#accuracy_svm_char_TFIDF_1 = train_model(svm.SVC(kernel = 'linear'), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars, valid_y)
#print(accuracy_svm_char_TFIDF_1)

# Random Forest
print('Random Forest')
accuracy_RF_count = train_model(RandomForestClassifier(), xtrain_count, train_y, xvalid_count, valid_y)
print(accuracy_RF_count)
accuracy_RF_TFIDF = train_model(RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
print(accuracy_RF_TFIDF)
accuracy_RF_ngram_TFIDF = train_model(RandomForestClassifier(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, valid_y)
print(accuracy_RF_ngram_TFIDF)
#accuracy_RF_char_TFIDF = train_model(RandomForestClassifier(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars, valid_y)
#print(accuracy_RF_char_TFIDF)






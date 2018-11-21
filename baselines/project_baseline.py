# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 11:27:40 2018

@author: jacky
"""
### Yelp dataset baseline models ###

##import libraries
import string
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import warnings

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


warnings.filterwarnings("ignore") 

##define helper functions

#tokenzie reviews 
def tokenize(reviewlist):
    return {i+1: word_tokenize(review) for i, review in enumerate(reviewlist)}
	#tokenizedwords = {}
	#for i,review in enumerate(reviewlist):
	#	tokenizedwords[i] = word_tokenize(review)
	#return tokenizedwords

#build a lexicon based on tokenzied word list
def createlexicon(tokenizedwords):
    lexicon = set()
    for i in range(1, len(tokenizedwords) + 1):
        lexicon.update(tokenizedwords[i])
    return lexicon

#calculate term frequency
def tf(word, tokenizedwords):
	return tokenizedwords.count(word)

#create tf-idf matrix based on words in lexicon
def TfIdf(tokenizedwords):
	lexicon = createlexicon(tokenizedwords)
	tfvector = {}
	for i in range(1, len(tokenizedwords) + 1):
		tfvector[i] = [tf(word, tokenizedwords[i]) for word in lexicon]
	return lexicon, tfvector

#create tags
def buildtags(dicts):
	tags = dicts.values()
	return tags

#make predictions based on training data (SVM-linear)
def classification(train_vecs, train_tags):
	#build a classifier and train it
    classif = OneVsRestClassifier(SVC(C = 1, kernel = "linear", gamma = 1, verbose = False, probability = False))
    classif.fit(train_vecs, train_tags)
    #use 5-fold CV to get prediction
    prediction = model_selection.cross_val_predict(classif, train_vecs, train_tags, cv = 5)
    print (accuracy_score(train_tags, prediction))
    print ("\n")
    print (classification_report(train_tags, prediction))
    print ("\n")
    print (confusion_matrix(train_tags, prediction))
    print ("\n")
    print (metrics.precision_score(train_tags, prediction, pos_label = None, average = 'weighted'))
    print ("\n")
    print (metrics.recall_score(train_tags, prediction, pos_label = None, average = 'weighted'))
    print ("\n")
    return

#make predictions based on training data (Naive Bayes)
def classification_1(train_vecs, train_tags):
    #build a classifier and train it
    classif_1 = MultinomialNB()
    classif_1.fit(train_vecs, train_tags)
    #use 5-fold CV to get prediction
    prediction_1 = model_selection.cross_val_predict(classif_1, train_vecs, train_tags, cv = 5)
    print (accuracy_score(train_tags, prediction_1))
    print ("\n")
    print (classification_report(train_tags, prediction_1))
    print ("\n")
    print (confusion_matrix(train_tags, prediction_1))
    print ("\n")
    print (metrics.precision_score(train_tags, prediction_1, pos_label = None, average = 'weighted'))
    print ("\n")
    print (metrics.recall_score(train_tags, prediction_1, pos_label = None, average = 'weighted'))
    print ("\n")
    return

#make predictions based on training data (Logistic Regression)
def classification_2(train_vecs, train_tags):
    #build a classifier and train it
    classif_2 = LogisticRegression()
    classif_2.fit(train_vecs, train_tags)
    #use 5-fold CV to get prediction
    prediction_2 = model_selection.cross_val_predict(classif_2, train_vecs, train_tags, cv = 5)
    print (accuracy_score(train_tags, prediction_2))
    print ("\n")
    print (classification_report(train_tags, prediction_2))
    print ("\n")
    print (confusion_matrix(train_tags, prediction_2))
    print ("\n")
    print (metrics.precision_score(train_tags, prediction_2, pos_label = None, average = 'weighted'))
    print ("\n")
    print (metrics.recall_score(train_tags, prediction_2, pos_label = None, average = 'weighted'))
    print ("\n")
    return

#make predictions based on training data (Random Forest)
def classification_4(train_vecs, train_tags):
    #build a classifier and train it
    classif_4 = RandomForestClassifier(n_estimators = 10, max_depth = None, random_state = None)
    classif_4.fit(train_vecs, train_tags)
    #use 5-fold CV to get prediction
    prediction_4 = model_selection.cross_val_predict(classif_4, train_vecs, train_tags, cv = 5)
    print (accuracy_score(train_tags, prediction_4))
    print ("\n")
    print (classification_report(train_tags, prediction_4))
    print ("\n")
    print (confusion_matrix(train_tags, prediction_4))
    print ("\n")
    print (metrics.precision_score(train_tags, prediction_4, pos_label = None, average = 'weighted'))
    print ("\n")
    print (metrics.recall_score(train_tags, prediction_4, pos_label = None, average = 'weighted'))
    print ("\n")
    return

#make predictions based on training data (SVM-poly)
def classification_5(train_vecs, train_tags):
	#build a classifier and train it
    classif_5 = OneVsRestClassifier(SVC(C = 1, kernel = "poly", gamma = 1, verbose = False, probability = False))
    classif_5.fit(train_vecs, train_tags)
    #use 5-fold CV to get prediction
    prediction_5 = model_selection.cross_val_predict(classif_5, train_vecs, train_tags, cv = 5)
    print (accuracy_score(train_tags, prediction_5))
    print ("\n")
    print (classification_report(train_tags, prediction_5))
    print ("\n")
    print (confusion_matrix(train_tags, prediction_5))
    print ("\n")
    print (metrics.precision_score(train_tags, prediction_5, pos_label = None, average = 'weighted'))
    print ("\n")
    print (metrics.recall_score(train_tags, prediction_5, pos_label = None, average = 'weighted'))
    print ("\n")
    return

#make predictions based on training data (SVM-rbf)
def classification_6(train_vecs, train_tags):
	#build a classifier and train it
    classif_6 = OneVsRestClassifier(SVC(C = 1, kernel = "rbf", gamma = 1, verbose = False, probability = False))
    classif_6.fit(train_vecs, train_tags)
    #use 5-fold CV to get prediction
    prediction_6 = model_selection.cross_val_predict(classif_6, train_vecs, train_tags, cv = 5)
    print (accuracy_score(train_tags, prediction_6))
    print ("\n")
    print (classification_report(train_tags, prediction_6))
    print ("\n")
    print (confusion_matrix(train_tags, prediction_6))
    print ("\n")
    print (metrics.precision_score(train_tags, prediction_6, pos_label = None, average = 'weighted'))
    print ("\n")
    print (metrics.recall_score(train_tags, prediction_6, pos_label = None, average = 'weighted'))
    print ("\n")
    return

#remove stopwords
def reStop(tokenizedwords):
    for a in range(1, len(tokenizedwords) + 1):
        filteredwords = [word for word in tokenizedwords[a] if word not in stopwords.words('english')]
        tokenizedwords[a] = filteredwords
    return tokenizedwords

##main function for multiple baseline models
def main():
    #loading data and initialization
    mydf = pd.read_csv("./business_reviews.tsv", delimiter="\t", encoding="utf-8")
    reviewlist = mydf['text']

    #call functions
    tokenizedwords = tokenize(reviewlist)
    lexicon, tfVector = TfIdf(tokenizedwords)
    tags = mydf['level'].apply(lambda x: int(x > 4))
    train_vecs = np.array(list(tfVector.values()))
    train_tags = np.array(list(tags))

    print ("SVM-linear")
    classification(train_vecs, train_tags) 
    print ("NB")
    classification_1(train_vecs, train_tags) 
    print ("LR-logistic")
    classification_2(train_vecs, train_tags) 
    print ("RF")
    classification_4(train_vecs, train_tags) 
    print ("SVM-poly")
    classification_5(train_vecs, train_tags) 
    print ("SVM-rbf")
    classification_6(train_vecs, train_tags) 
    print ("\n")
    print ("remove stopwords then train again")    
    print ("\n")
    tokenizedwords = reStop(tokenizedwords)
    lexicon, tfVector = TfIdf(tokenizedwords)
    train_vecs = np.array(list(tfVector.values()))
    train_tags = np.array(list(tags))    
    print ("SVM-linear")
    classification(train_vecs, train_tags) 
    print ("NB")
    classification_1(train_vecs, train_tags) 
    print ("LR-logistic")
    classification_2(train_vecs, train_tags) 
    print ("RF")
    classification_4(train_vecs, train_tags) 
    print ("SVM-poly")
    classification_5(train_vecs, train_tags) 
    print ("SVM-rbf")
    classification_6(train_vecs, train_tags) 
    print ("\n")
    print ("5-class Classification based on stars")    
    print ("\n")

    train_tags = np.array(list(tags))
    print ("SVM-linear")
    classification(train_vecs, train_tags) 
    print ("NB")
    classification_1(train_vecs, train_tags) 
    print ("LR-logistic")
    classification_2(train_vecs, train_tags) 
    print ("RF")
    classification_4(train_vecs, train_tags) 
    print ("SVM-poly")
    classification_5(train_vecs, train_tags) 
    print ("SVM-rbf")
    classification_6(train_vecs, train_tags) 

if __name__ == "__main__":
    main()
    

    
    
    
    
    
    
        
 
    
    
    
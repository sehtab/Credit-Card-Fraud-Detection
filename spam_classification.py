#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 19:46:39 2021

@author: altair
"""

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import log, sqrt
import pandas as pd
import numpy as np
import string

df = pd.read_csv('emails.csv', encoding= 'latin-1')
print(df.head())

# print the shape
print(df.shape)

# get column names
print(df.columns)

# check for duplicates and remove
df.drop_duplicates(inplace = True)

# check the shape
print(df.shape)

# check number of missing data
print(df.isnull().sum())

def process_text(text):
    #1 
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    #2
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    #3
    return clean_words

# show the tokenization
print(df['text'].head().apply(process_text))

# convert a collection of text to a matrix
from sklearn.feature_extraction.text import CountVectorizer
messages_bow = CountVectorizer(analyzer=process_text).fit_transform(df['text'])

# train-test-split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(messages_bow, df['spam'], test_size=0.25, random_state=0)

# get the shape of bow
print(messages_bow.shape)

# create and train Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(x_train, y_train)

# print the prediction
print(classifier.predict(x_train))

# print the actual values
print(y_train.values)

# evaluate the model on training data set
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred = classifier.predict(x_train)
print(classification_report(y_train, pred))
print()
print('Confusion Matrix: \n', confusion_matrix(y_train, pred))
print()
print('Accuracy:', accuracy_score(y_train, pred))

# print the prediction
print(classifier.predict(x_test))

# print the actual values
print(y_test.values)   

# evaluate the model on training data set
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred = classifier.predict(x_test)
print(classification_report(y_test, pred))
print()
print('Confusion Matrix: \n', confusion_matrix(y_test, pred))
print()
print('Accuracy:', accuracy_score(y_test, pred)) 
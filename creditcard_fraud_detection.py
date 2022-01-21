#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 00:30:57 2021

@author: altair
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from scipy.stats import norm
from scipy.stats import multivariate_normal
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('creditcard.csv')
print(df.head())
print(df.shape)
print(df.describe())
print('Missing Values:', df.isnull().values.any())

#plot normal & fraud
count_classes = pd.value_counts(df['Class'], sort=True)
count_classes.plot(kind='bar', rot=0)
plt.title('Distributed Transactions')
plt.xticks(range(2), ['Normal', 'Fraud'])
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

# heatmap
sns.heatmap(df.corr(), vmin=-1)
plt.show()

############################
# functions
############################
def train_validation_splits(df):
    # fraud transaction
    fraud = df[df['Class'] == 1]
    # normal transactins
    normal = df[df['Class'] == 0]
    print('normal:', normal.shape[0])
    print('fraud:', fraud.shape[0])
    
    normal_test_start = int(normal.shape[0] * 0.2)
    fraud_test_start = int(fraud.shape[0] * 0.5)
    normal_train_start = normal_test_start * 2
    
    val_normal = normal[:normal_test_start]
    val_fraud = fraud[:fraud_test_start]
    validation_set = pd.concat([val_normal, val_fraud], axis=0)
    
    test_normal = normal[normal_test_start:normal_train_start]
    test_fraud = fraud[fraud_test_start: fraud_test_start:fraud.shape[0]]
    test_set = pd.concat([test_normal, test_fraud], axis=0)
    
    xval = validation_set.iloc[:,:-1]
    yval = validation_set.iloc[:,-1]
    
    xtest = test_set.iloc[:,:-1]
    ytest = test_set.iloc[:, :-1]
    
    train_set = normal[normal_train_start:normal.shape[0]]
    xtrain = train_set.iloc[:, :-1]
    
    return xtrain.to_numpy(), xtest.to_numpy(), xval.to_numpy(), ytest.to_numpy(), yval.to_numpy()

def estimate_gaussian_params(x):
    """ calculate the mean and the covariance for each feature 
    Arguments:
        x: dataset
        """
    mu = np.mean(x, axis=0)
    sigma = np.cov(x.T)
    return mu, sigma

def metrics(y, predictions):
    fp = np.sum(np.all([predictions == 1, y == 0], axis=0))
    tp = np.sum(np.all([predictions == 1], axis=0))
    fn = np.sum(np.all([predictions == 0, y == 1], axis=0))
    
    precision = (tp / (tp + fp)) if (tp+fp) > 0 else 0
    recall = (tp/ (tp + fn)) if (tp+fn) > 0 else 0
    f1 = (2*precision*recall) / (precision + recall) if (precision+recall) > 0 else 0
    return precision, recall, f1
def selectthreshold(yval, pval):
    e_values = pval
    bestf1 = 0
    bestepsilon = 0
    
    for epsilon in e_values:
        predictions = pval < epsilon
        (precision, recall, f1) = metrics(yval, predictions)
        if f1 > bestf1:
            bestf1 = f1
            bestepsilon = epsilon
    return bestepsilon, bestf1

#############################

fig, axs = plt.subplots(6, 5, squeeze= False)
for i, ax in enumerate(axs.flatten()):
    ax.set_facecolor('xkcd:charcoal')
    ax.set_title(df.columns[i])
    sns.distplot(df.iloc[:, i], ax=ax, fit=norm, color='#DC143C', fit_kws={'color': '#4e8ef5'})
    ax.set_xlabel('')
fig.tight_layout(h_pad=-1.5, w_pad=-1.5)
plt.show()

classes = df['Class']
df.drop(['Time', 'Class', 'Amount'], axis=1, inplace=True)
cols = df.columns.difference(['Class'])
mmscaller = MinMaxScaler()
df = mmscaller.fit_transform(df)
df = pd.DataFrame(data=df, columns= cols)
df = pd.concat([df, classes], axis=1)

(xtrain, xtest, xval, ytest, yval) = train_validation_splits(df)
(mu, sigma) = estimate_gaussian_params(xtrain)

# calculate gaussian pdf
p = multivariate_normal.pdf(xtrain, mu, sigma)
pval = multivariate_normal.pdf(xval, mu, sigma)
ptest = multivariate_normal.pdf(xtest, mu, sigma)

(epsilon, f1) = selectthreshold(yval, pval)
print('Best epsilon found:', epsilon)
print('Best F1 on cross validation set:', f1)

(test_precision, test_recall, test_f1) = metrics(ytest, ptest<epsilon)
print('Outliers found:', np.sum(ptest < epsilon))
print('Test set precision:', test_precision)
print('Test set Recall:', test_recall)
print('Test set F1 score:', test_f1)

 
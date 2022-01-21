#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:19:37 2021

@author: altair
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

df = pd.read_csv('creditcard.csv')
print(df.head())
print(df.shape)
print(df.describe())
print('Missing Values:', df.isnull().values.any())

# determine fraud caes in dataset
fraud = df.loc[df['Class'] == 1]
normal = df.loc[df['Class'] == 0]
print(fraud, normal)
print(fraud.sum())
sns.relplot(x='Amount', y='Time', hue= 'Class' , data= df)

from sklearn import linear_model
from sklearn.model_selection import train_test_split
x = df.iloc[:, :-1]
y = df['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)
clf= linear_model.LogisticRegression(C=1e5)
clf.fit(x_train, y_train)

y_pred = np.array(clf.predict(x_test))
y = np.array(y_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print('\nConfusion Matrix:', confusion_matrix(y_test, y_pred))
print('\nAccuracy Score:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
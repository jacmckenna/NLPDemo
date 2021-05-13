# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:02:10 2021

@author: jacmckenna
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import csv
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


with open('Intentsubset1.csv', 'r') as fd:
    reader = csv.DictReader(fd)
    labeledExampleUtteranceWithMLEntity = list(reader)
df = pd.DataFrame(labeledExampleUtteranceWithMLEntity)
X = df['text']
Y = df['intentName']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify = Y)


Y = np.array(Y_train)

vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(X_train).toarray()

clf = RandomForestClassifier()
clf.fit(X, Y)



Predictions = clf.predict(vectorizer.transform(X_test))
print(classification_report(Y_test,Predictions))
conf_mat = confusion_matrix(Y_test, Predictions)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=list(set(Y_test)), yticklabels=list(set(Y_test)))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


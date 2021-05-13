# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 17:06:47 2021

@author: jacmckenna
"""

from functools import reduce
import csv
import json, time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

with open('Intentsubset1.csv', 'r') as fd:
    reader = csv.DictReader(fd)
    labeledExampleUtteranceWithMLEntity = list(reader)
df = pd.DataFrame(labeledExampleUtteranceWithMLEntity)
X = df['text']
Y = df['intentName']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify = Y)


with open('prediction_results.csv', 'r') as fd2:
    reader = csv.DictReader(fd2)
    Predictions = list(fd2)
    Predictions_strip =[]
    for x in Predictions:
        x_new = x.rstrip()
        Predictions_strip.append(x_new)
        
print(classification_report(Y_test,Predictions_strip))
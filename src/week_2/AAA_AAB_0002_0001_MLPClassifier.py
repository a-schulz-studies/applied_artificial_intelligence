#                                       iA
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

csvFile = pd.read_csv('winequality-red.csv')

# print(csvFile)

colNames = list(csvFile.columns)

print(colNames)
# x = csvFile[]

colNamesX = colNames.copy()
colNamesX.remove('quality')
dfX = csvFile[colNamesX]
X = dfX.values
Y = csvFile['quality'].values

print(set(Y))

numsMe = np.random.permutation(csvFile.shape[0])
_thresh = int(np.floor(csvFile.shape[0]*.8))

print(numsMe)
print(_thresh)

trainInds =  numsMe[:_thresh]
testInds  =  numsMe[_thresh:]

TrainX, TestX = X[trainInds], X[testInds]
TrainY, TestY = Y[trainInds], Y[testInds]

clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(12, ), random_state=1)

clf.fit(TrainX, TrainY)

preds = clf.predict(TestX)

cm = confusion_matrix(TestY, preds)
print(cm)

acc = accuracy_score(TestY, preds)
print(acc)

accb = balanced_accuracy_score(TestY, preds)
print(accb)

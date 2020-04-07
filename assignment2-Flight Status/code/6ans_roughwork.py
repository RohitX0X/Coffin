# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 21:48:13 2020

@author: Rohit
"""

import numpy as np
import sys
import functions as fn
import matplotlib.pyplot as plt
import pandas as pd
teseee=fn.htheta(np.array([[0],[0]]),np.array([0,0]))
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
# Importing the dataset
dataset = pd.read_csv('../FlightDelays.csv')
dataset=dataset.to_numpy()

X = dataset[:, 0:13]
y = dataset[:, 12]

#plt.scatter(X[:,8],y)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])



labelencoder_X_2 = LabelEncoder()
X[:, 3] = labelencoder_X_2.fit_transform(X[:, 3])
labelencoder_X_1 = LabelEncoder()
X[:, 8] = labelencoder_X_1.fit_transform(X[:, 8])
labelencoder_X_1 = LabelEncoder()
X[:, 5] = labelencoder_X_1.fit_transform(X[:, 5])
labelencoder_X_1 = LabelEncoder()
X[:, 12] = labelencoder_X_1.fit_transform(X[:, 12])
labelencoder_X_1 = LabelEncoder()
X[:, 11] = labelencoder_X_1.fit_transform(X[:, 11])
labelencoder_X_1 = LabelEncoder()
X[:, 7] = labelencoder_X_1.fit_transform(X[:, 7])
idx=np.where(y==0)
ad=np.where(X[:,1]==7)
times=np.array(X[:,0]).reshape(2201,1)
times=np.int32(dads).flatten()
times=(times-600)//88.88
carrier=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
days=[0,0,0,0,0,0]
idx=np.array(idx).flatten()
for i in idx:
    
    a=times[i]
    carrier[a]+=1
    


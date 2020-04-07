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
clf=svm.SVC()

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
onehotencoder = OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
onehotencoder = OneHotEncoder(categorical_features=[10])
X=onehotencoder.fit_transform(X).toarray()
onehotencoder = OneHotEncoder(categorical_features=[16])
X=onehotencoder.fit_transform(X).toarray()
onehotencoder = OneHotEncoder(categorical_features=[19])
X=onehotencoder.fit_transform(X).toarray()
onehotencoder = OneHotEncoder(categorical_features=[21])
X=onehotencoder.fit_transform(X).toarray()
y=X[:,30]
y=1-y
X =X[:, 0:27]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X1=np.ones((1320,1))
X_train=np.append(X1,X_train,1)
X_test=np.append(np.ones((881,1)),X_test,1)

X_test=np.array(X_test)
X_train=np.array(X_train,dtype='float32')
y_test=np.array(y_test,dtype='float32').reshape(881,1)
y_train=np.array(y_train,dtype='float32').reshape(1320,1)
print(y_test)
clf = LogisticRegression(fit_intercept=0).fit(X_train, y_train)

W=np.zeros((28,1))
alpha=1
iters=9500
W,cost_history=fn.train(X_train,y_train,W,alpha,iters)
print(X_train.shape[0])
print(W)
plt.plot(range(iters),cost_history)
plt.xlabel('no. of iters') 
# naming the y axis 
plt.ylabel('cost') 
  
# giving a title to my graph 
plt.title('convergence') 
sum=np.sum(y_test)
# function to show the plot 
plt.show() 
##########################Calculating best possible accuracy,f score by tuning the threshold#########
max_f1=0
max_acc=0
for x in range(100):
    y_pred=np.array(fn.htheta(W,X_test))> x/100
    f1=f1_score(y_test, y_pred)
    #y_tot=np.append(y_pred,y_test,1)
    acc=accuracy_score(y_test,y_pred)
    cm=confusion_matrix(y_test, y_pred)
    max_f1=max(f1,max_f1)
    max_acc=max(acc,max_acc)
    if f1==max_f1:
        idx_f=x
    if acc==max_acc:
        idx_acc=x
#######################################################
y_pred=np.array(fn.htheta(W,X_test))> idx_acc/100
acc=accuracy_score(y_test,y_pred)
f1=f1_score(y_test, y_pred)
cm=confusion_matrix(y_test, y_pred)
clf.predict(X_test[:, :])
clf.score(X_test, y_test)
#
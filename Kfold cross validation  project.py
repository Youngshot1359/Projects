# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 20:44:59 2021

@author: marvin

The goal of this code is to explore performing cross validation on data
with multiple linear regression models.

This code is provided to get you started quickly. 

Review/excute each section and be sure to understand before doing assignment.
To do the assignment, just extract whatver code you need to complete the 
assignment. Feel free to use other code you may come across on the web or
 other. Just be sure you understand it.


"""

##############################################################################
# read in the data / template section 1

import pandas as pd


df = pd.read_csv('/Users/phenom/Downloads/wdbc.data', header=None)
df.head()
df.shape

# The cancer data file has 30 features.
# Assign the 30 features to a numpy array x.
# Then using a lable encoder object, transform the class labels
# in column 2 (M for malignent / B for benign) into integers
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder

# get 30 features
X = df.loc[:, 2:2].values
#get lables
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)


# List the label for each class.
le.classes_

# To determine which value each class mapped into
# mapping: M mapped to 1, B mapped to 0
le.transform(['M','B'])


##############################################################################
# Training/Validation/Test procedure  / template section 2
# divide the data into 80% training and 20% test
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                     test_size=0.80,
                     stratify=y,
                     random_state=1)


# run individually to see split sizes of the data
X.shape
X_train.shape
X_test.shape

#now develop model parameters/hyperparameters with train/validate
# then estimate the modesl generalization performance on the test set

##############################################################################
# Scaling the data / template section 3
# you may or may not need to scale data. Depends on the data.
# this section raise important issue about how to compute performance
# accuracies for regression classfier algorithms. Also, note that
# linear regression is not the best case for classfication, as is being used
# in this assignment. Logistic Regression is a better option which we will
# cover later.
#Linear

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Training set results
y_pred = regressor.predict(X_train)
#import matplotlib.pyplot as plt
#plt.figure(0)
#plt.plot(y_pred)

#regression returns a real value, so convert it to binary values.
y_pred=(y_pred > 0.5).astype(int)

TP=np.sum(y_pred==y_train)
accuracy=TP/y_pred.size
print('Train Accuracy: %.3f' % accuracy)
print('Train Accuracy (coeff of determination): %.3f' % regressor.score(X_train, y_train))

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred=(y_pred > 0.5).astype(int)
TP=np.sum(y_pred==y_test)
accuracy=TP/y_pred.size
print('Test Accuracy: %.3f' % accuracy)
print('Test Accuracy (coeff of determination): %.3f' % regressor.score(X_test, y_test))

##############################################################################
# k-fold Linear Regression/ 
import numpy as np
from sklearn.model_selection  import StratifiedKFold
S_KFold = StratifiedKFold(n_splits=10)
kfold = S_KFold.split(X_train, y_train)

scores = []
accuracies = []
for k, (train, test) in enumerate(kfold):    
    
    regressor.fit(X_train[train], y_train[train])
    
#compute  the regression error    
    score = regressor.score(X_train[test], y_train[test])        
    scores.append(score)                
    print('Fold Linear: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
          np.bincount(y_train[train]), score))

# compute true classification prediction accuracies    
    y_pred = regressor.predict(X_train[test]) # prediction return reals
    y_pred=(y_pred > 0.5).astype(int) #convert to discrete 0 or 1 for malignenat or benign
    TP=np.sum(y_pred==y_train[test]) #compute number of true positives
    accuracy=TP/y_pred.size
    accuracies.append(accuracy)
    print('  Fold: Test Accuracy: %.3f' % accuracy)
                
    
print('\nCV accuracy (coeff of determination): %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(accuracies), np.std(accuracies))) #use this CV error!

# do the above k-fold algorithm for differnt algorithms.
# Then choose the model that gives the best CV error.
# then use that model, train it on all the train & validation data,
# then use the test data to check generlization performance.
# in the above, use the accuracies to determine performance.

##############################################################################
# confusion matrix  Linear Regression









# using the test data, generate a confusion matrix to view the performance
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                     test_size=0.80,
                     stratify=y,
                     random_state=1)

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Training set results
y_pred = regressor.predict(X_test)

#regression returns a real value, so convert it to binary values.
y_pred=(y_pred > 0.5).astype(int)

from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

#now use matplot lib to give a nice plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)

for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('')
plt.ylabel('')
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
plt.tight_layout()






plt.xlabel('Prediction')
plt.ylabel('Truth')
#plt.savefig('images/06_09.png', dpi=300)
plt.show()

#####################################################
#Naive Bayes Scaling and Prediction 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                     test_size=0.80,
                     stratify=y,
                     random_state=1)

from sklearn.naive_bayes import GaussianNB
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
nb = GaussianNB()
nb.fit(X_train, y_train)
GaussianNB()





# Predicting the Training set results
y_pred = nb.predict(X_train)
y_pred = nb.predict(X_test)
y_pred=(y_pred > 0.5).astype(int)
TP=np.sum(y_pred==y_train)
accuracy=TP/y_pred.size
print('Train Accuracy Naive Bayes: %.3f' % accuracy)
print('Train Accuracy Naive Bayes(coeff of determination): %.3f' %
nb.score(X_train, y_train))
y_pred = nb.predict(X_test)

#Test Predication Results
y_pred = nb.predict(X_test)
y_pred=(y_pred > 0.5).astype(int)
TP=np.sum(y_pred==y_test)
accuracy=TP/y_pred.size
print('Test Accuracy Naive Bayes: %.3f' % accuracy)
print('Test Accuracy Naive Bayes(coeff of determination): %.3f' %
nb.score(X_test, y_test))
print('\n')

############################################################
#Naive Kfolds 


from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
S_KFold = StratifiedKFold(n_splits=10)
kfold = S_KFold.split(X_train, y_train)
scores = []
accuracies = []
for k, (train, test) in enumerate(kfold):
#Naive Regresion Error
 nb.fit(X_train[train], y_train[train])
 score = nb.score(X_train[test], y_train[test])
 scores.append(score)
 print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
 np.bincount(y_train[train]), score))
 
 # compute true classification prediction accuracies
 y_pred = nb.predict(X_train[test]) # prediction return reals
 y_pred=(y_pred > 0.5).astype(int) #convert to discrete 0 or 1 for malignenat or benign
 TP=np.sum(y_pred==y_train[test]) #compute number of true positives
 accuracy=TP/y_pred.size
 accuracies.append(accuracy)
 print('  Fold: Test Accuracy: %.3f' % accuracy)

print('\nCV accuracy Naive Bayes (coeff of determination): %.3f +/- %.3f'
% (np.mean(scores), np.std(scores)))
print('\nCV accuracy Naive Bayes: %.3f +/- %.3f' % (np.mean(accuracies),
np.std(accuracies)))

################################################################
#Naive Confusion Matrix

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                     test_size=0.80,
                     stratify=y,
                     random_state=1)
# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predicting the Training set results
y_pred = nb.predict(X_test)
# returns a real value, so convert it to binary values.
y_pred=(y_pred > 0.5).astype(int)
from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)
#now use matplot lib to give a nice plot
import seaborn as sns
ax = sns.heatmap(confmat/np.sum(confmat), annot=True, 
            fmt='.2%', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\n Naive Predicted Values')
ax.set_ylabel('Naive Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()

###################################################################
#K Nearest Neighbor

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                     test_size=0.80,
                     stratify=y,
                     random_state=1)
from sklearn.neighbors import KNeighborsClassifier


kn = KNeighborsClassifier()
kn.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred=(y_pred > 0.5).astype(int)
TP=np.sum(y_pred==y_train)
accuracy=TP/y_pred.size
print('\n')
print('Train Accuracy KNN: %.3f' % accuracy)
print('Train Accuracy KNN(coeff of determination): %.3f' %
kn.score(X_train, y_train))
# Predicting the Test set results
y_pred = kn.predict(X_test)
y_pred=(y_pred > 0.5).astype(int)
TP=np.sum(y_pred==y_test)
accuracy=TP/y_pred.size
print('Test Accuracy KNN: %.3f' % accuracy)
print('Test Accuracy KNN(coeff of determination): %.3f' %
kn.score(X_test, y_test))
print('\n')
######################################################
#KNN Kfold
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
S_KFold = StratifiedKFold(n_splits=10)
kfold = S_KFold.split(X_train, y_train)
scores = []
accuracies = []
for k, (train, test) in enumerate(kfold):

 kn.fit(X_train[train], y_train[train])

#compute the regression error
 score = kn.score(X_train[test], y_train[test])
 scores.append(score)
 print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
 np.bincount(y_train[train]), score))
# compute true classification prediction accuracies
 y_pred = kn.predict(X_train[test]) # prediction return reals
 y_pred=(y_pred > 0.5).astype(int) #convert to discrete 0 or 1 for malignenat or benign
 TP=np.sum(y_pred==y_train[test]) #compute number of true positives
 accuracy=TP/y_pred.size
 accuracies.append(accuracy)
 print('  Fold: Test Accuracy: %.3f' % accuracy)#

print('\nCV accuracy KNN (coeff of determination): %.3f +/- %.3f' %
(np.mean(scores), np.std(scores)))
print('\nCV accuracy KNN: %.3f +/- %.3f' % (np.mean(accuracies),
np.std(accuracies)))
########################################
#KNN Confusion Matrix
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                     test_size=0.80,
                     stratify=y,
                     random_state=1)
# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
kn = KNeighborsClassifier()
kn.fit(X_train, y_train)
# Predicting the Training set results
y_pred = kn.predict(X_test)
#returns a real value, so convert it to binary values.
y_pred=(y_pred > 0.5).astype(int)
from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

#Displaying Data
import seaborn as sns
ax = sns.heatmap(confmat/np.sum(confmat), annot=True, 
            fmt='.2%', cmap='Greens')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\n KNeigbor Predicted Values')
ax.set_ylabel('KNeigbor Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()

###############################################
#Logistic Regression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                     test_size=0.80,
                     stratify=y,
                     random_state=1)

from sklearn.linear_model import LogisticRegression
# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



logR= LogisticRegression()
logR.fit(X_train, y_train)
LogisticRegression()


y_pred = logR.predict(X_test)
y_pred=(y_pred > 0.5).astype(int)
TP=np.sum(y_pred==y_train)
accuracy=TP/y_pred.size
print('Train Accuracy Logistic Regression: %.3f' % accuracy)
print('Train Accuracy Logistic Regression(coeff of determination): %.3f' %
logR.score(X_train, y_train))
y_pred = logR.predict(X_test)

#Test Predication Results
y_pred = logR.predict(X_test)
y_pred=(y_pred > 0.5).astype(int)
TP=np.sum(y_pred==y_test)
accuracy=TP/y_pred.size
print('Test Accuracy Logistic Regression Bayes: %.3f' % accuracy)
print('Test Accuracy Logistic Regression(coeff of determination): %.3f' %
logR.score(X_test, y_test))
print('\n')
############################################
#Logistic Kfolds 
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
S_KFold = StratifiedKFold(n_splits=10)
kfold = S_KFold.split(X_train, y_train)
scores = []
accuracies = []

for k, (train, test) in enumerate(kfold):
#Logistic Regresion Error
 logR.fit(X_train[train], y_train[train])
 score = logR.score(X_train[test], y_train[test])
 scores.append(score)
 print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
 np.bincount(y_train[train]), score))
 
 # compute true classification prediction accuracies
 y_pred = logR.predict(X_train[test]) # prediction return reals
 y_pred=(y_pred > 0.5).astype(int) #convert to discrete 0 or 1 for malignenat or benign
 TP=np.sum(y_pred==y_train[test]) #compute number of true positives
 accuracy=TP/y_pred.size
 accuracies.append(accuracy)
 print('  Fold: Test Accuracy: %.3f' % accuracy)

print('\nCV accuracy Logistic Regression (coeff of determination): %.3f +/- %.3f'
% (np.mean(scores), np.std(scores)))
print('\nCV accuracy Logistic Regression: %.3f +/- %.3f' % (np.mean(accuracies),
np.std(accuracies)))


#####################################################
#Logistic Regression Confusion Matrix
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                     test_size=0.80,
                     stratify=y,
                     random_state=1)

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


logR.fit(X_train, y_train)

# Predicting the Training set results
y_pred = logR.predict(X_test)

#regression returns a real value, so convert it to binary values.
y_pred=(y_pred > 0.5).astype(int)

from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

#Display Data
import seaborn as sns
ax = sns.heatmap(confmat/np.sum(confmat), annot=True, 
            fmt='.2%', cmap='Reds')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\n Logistic Predicted Values')
ax.set_ylabel('Logistic Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()


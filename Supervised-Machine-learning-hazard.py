from tkinter.messagebox import NO
from turtle import TPen
from matplotlib.pyplot import clf
import numpy as np
import pandas as pd 

dataset = pd.read_csv('neo.csv', sep=',')

# count used to check if orbiting_body is all same string of 'Earth' and sentry_object is string 'False'
count = (dataset.iloc[:,-4])
count2 = (dataset.iloc[:,-3])

# normalising and removing names of asteroids and repeating uneeded values for ML
dataset = dataset.drop(['name', 'orbiting_body', 'sentry_object'], axis=1)

# check dataset has been loaded and columns dropped correctly
# print(dataset)

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# select all but last column for X and last for y
X = dataset[dataset.columns[0:-1]]
y = dataset[dataset.columns[-1]]

# make sure correct columns are chosen
# print(X)
# print(y)

# using train_test_split import to split the dataset with a randomness of 30
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=None)
print(Xtest)

# create default MLPClassifier
clasi = MLPClassifier()

# fit and train the MLPClassifier on the Xtrain and ytrain data
clasi.fit(Xtrain, ytrain)

# predict and store results of predictions against Xtest data by the trained MLPClassifier
ypred = clasi.predict(Xtest)

# import confusion matrix
from sklearn.metrics import confusion_matrix
# comparing the predictions against the actual results in ytest
cm = confusion_matrix(ypred, ytest)

# check confusion matrix is working
print(cm)

# stores each sqaure in confusion matrix in a variable to use for formula
TP = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TN = cm[1,1]

# formula used to get accuracy of the MLPClassifier
accuracy = (TP + TN) / (TP + TN + FP + FN)

# printing the accuracy 
print("Accuracy of MLPClassifier : ", accuracy)

# enter the following statistics of a meteor for MLP to predict if it is hazardous or not:
elements = {
        'id':  [2277475],
        'est_diameter_min': [0.2658],
        'est_diameter_max': [0.5943468684],
        'relative_velocity': [73588.7266634981],
        'miss_distance': [61438126.52395093],
        'absolute_magnitude': [20.0],
        }

elements = pd.DataFrame(elements)
print("True/False the entered asteroid is hazardous... ", clasi.predict(elements))

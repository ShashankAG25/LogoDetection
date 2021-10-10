from skimage import exposure, feature
import numpy as np
import cv2 as cv
import glob
import os
import pickle
from sklearn import metrics
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# Get Paths

TrainedModel = loaded_model = pickle.load(open("Models/TrainedModel.yml", 'rb'))

dataset = load_wine()
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
knn = KNeighborsClassifier(1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(y_train)
print(y_pred)

cm= metrics.confusion_matrix(y_test, y_pred)
print(cm)
# Printing the precision and recall, among other metrics
print(metrics.classification_report(y_test, y_pred,labels=None))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
disp.plot()
disp.ax_.set(title='Sklearn Confusion Matrix with labels!!', xlabel =" x axis" , ylabel ="y axis")
plt.show()
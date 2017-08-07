#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

### make sure you use // when dividing for integer division


features_train = features_train[:len(features_train)//100] 
labels_train = labels_train[:len(labels_train)//100] 



from sklearn.svm import SVC  
import numpy as np
clf = SVC(kernel="rbf", C=10000.0)
t0 = time()
clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")

t0 = time()
predictions = clf.predict(features_test)
unique, counts = np.unique(predictions, return_counts=True)
print(dict(zip(unique, counts)))


print ("Predicting time:", round(time()-t0, 3), "s")

print ("Score: ---- : ", clf.score(features_test, labels_test))
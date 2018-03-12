from time import time
from email_preprocess import preprocess

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

from sklearn.tree import DecisionTreeClassifier
import numpy as np
clf = DecisionTreeClassifier(min_samples_split=40)
t0 = time()
print("Le", len(features_train[0]))

clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")
t0 = time()
predictions = clf.predict(features_test)
unique, counts = np.unique(predictions, return_counts=True)
print(dict(zip(unique, counts)))


print ("Predicting time:", round(time()-t0, 3), "s")

print ("Score: ---- : ", clf.score(features_test, labels_test))


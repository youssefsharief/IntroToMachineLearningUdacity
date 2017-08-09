#!/usr/bin/python3

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


from sklearn.neighbors import KNeighborsClassifier

import numpy as np
from time import time
import sys
sys.path.append("../tools/")

clf = KNeighborsClassifier(n_neighbors=4)
t0 = time()
clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")

t0 = time()
predictions = clf.predict(features_test)
# unique, counts = np.unique(predictions, return_counts=True)
# print(dict(zip(unique, counts)))


print ("Predicting time:", round(time()-t0, 3), "s")

print ("Score: ---- : ", clf.score(features_test, labels_test))







try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass

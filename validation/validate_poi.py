  


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""
from sklearn.cross_validation import train_test_split
import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.tree import DecisionTreeClassifier
import numpy as np


data_dict = pickle.load(open("../final_project/final_project_dataset_unix2.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)


clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
print(clf.score(features_train, labels_train))
print(clf.score(features_test, labels_test))
### it's all yours from here forward!  



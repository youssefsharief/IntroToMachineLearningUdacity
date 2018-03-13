  


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, precision_score

def get_data_and_format():
    data_dict = pickle.load(open("../final_project/final_project_dataset_unix2.pkl", "rb") )
    ### add more features to features_list!
    features_list = ["poi", "salary"]
    data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson14_keys2.pkl')
    labels, features = targetFeatureSplit(data)
    return features, labels


def fit_and_predict(features_train, labels_train, features_test,labels_test):
    clf = DecisionTreeClassifier()
    clf.fit(features_train, labels_train)
    print("Accuracy score:", clf.score(features_test,labels_test))
    preds = clf.predict(features_test)
    return preds


def no_of_pois(preds, labels_test):
    num_pois = len([pred for pred in preds if pred == 1])
    print("Number of POIs predicted:", num_pois)
    print("Total number of people in test set:", len(preds))
    print("Recall Score:", recall_score(labels_test, preds))
    print("Precision Score:", precision_score(labels_test, preds))
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred, actual in zip(preds, labels_test):
        if pred == actual and actual == 1:
            true_positives += 1
        elif pred == 1 and actual == 0:
            false_positives += 1
        elif pred == 0 and actual == 1:
            false_negatives += 1

    print("True Positives:", true_positives)
    print("False Positives", false_positives)
    print("False Negatives", false_negatives)

    print("\nPrecision (POIs):", true_positives/(true_positives + false_positives))
    print("Recall (POIs)", true_positives/(true_positives + false_negatives))


def main():
    features, labels = get_data_and_format()
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 42)
    preds = fit_and_predict(features_train, labels_train, features_test, labels_test)
    no_of_pois(preds, labels_test)

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
no_of_pois(predictions, true_labels)
  

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
import matplotlib.pyplot as plt
import pprint

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = ['poi', 'salary', 'total_payments', 'bonus', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock']
#'email_address',
email_features = ['poi', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset_unix2.pkl", "rb") )

### Task 2: Remove outliers
financial_outliers = featureFormat(data_dict, financial_features)
email_outliers = featureFormat(data_dict, email_features)

def remove_ouliers():
    identified_outliers = ["TOTAL", "LAVORATO JOHN J", "MARTIN AMANDA K", "URQUHART JOHN A", "MCCLELLAN GEORGE",
                           "SHANKMAN JEFFREY A", "WHITE JR THOMAS E", "PAI LOU L", "HIRKO JOSEPH"]
    for outlier in identified_outliers:
        data_dict.pop(outlier)

def get_outlier(feature, value):
    for name, features in data_dict.items():
        if features[feature] == value:
            print("Outlier is:", name, features['poi'])

get_outlier('to_messages', 15149)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
# estimators = [('reduce_dim', PCA(n_components=4)), ('nb', GaussianNB())]
estimators = [('reduce_dim', PCA(n_components=2)), ('svm', SVC())]
# clf = Pipeline(estimators)
clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.

# ### Task 5: Tune your classifier to achieve better than .3 precision and recall
# ### using our testing script. Check the tester.py script in the final project
# ### folder for details on the evaluation method, especially the test_classifier
# ### function. Because of the small size of the dataset, the script uses
# ### stratified shuffle split cross validation. For more info:
# ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# test_classifier(clf, my_dataset, features_list)
#
# # Example starting point. Try investigating other evaluation techniques!
# # from sklearn.cross_validation import train_test_split
# # features_train, features_test, labels_train, labels_test = \
# #     train_test_split(features, labels, test_size=0.3, random_state=42)
#
# ### Task 6: Dump your classifier, dataset, and features_list so anyone can
# ### check your results. You do not need to change anything below, but make sure
# ### that the version of poi_id.py that you submit can be run on its own and
# ### generates the necessary .pkl files for validating your results.
#
# dump_classifier_and_data(clf, my_dataset, features_list)

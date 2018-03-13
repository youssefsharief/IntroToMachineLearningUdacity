  

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from poi_id_pca import determine_the_optimum_number_of_pac_components
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
feature_list = ['poi', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'salary', 'total_payments', 'bonus', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset_unix2.pkl", "rb") )

### Task 2: Remove outliers
data_dict.pop('TOTAL')

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB

logistic = LogisticRegression()
clf = DecisionTreeClassifier(min_samples_split=14, random_state=2)
linear = LinearRegression()
naive = GaussianNB()
from sklearn.svm import SVC
svm = SVC(kernel="rbf")

determine_the_optimum_number_of_pac_components(clf, features, labels)


# ### Task 5: Tune your classifier to achieve better than .3 precision and recall
# ### using our testing script. Check the tester.py script in the final project
# ### folder for details on the evaluation method, especially the test_classifier
# ### function. Because of the small size of the dataset, the script uses
# ### stratified shuffle split cross validation. For more info:
# ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# test_classifier(clf, my_dataset, features_list)
#
# # Example starting point. Try investigating other evaluation techniques!



#
# ### Task 6: Dump your classifier, dataset, and features_list so anyone can
# ### check your results. You do not need to change anything below, but make sure
# ### that the version of poi_id.py that you submit can be run on its own and
# ### generates the necessary .pkl files for validating your results.
#
dump_classifier_and_data(clf, my_dataset, feature_list)

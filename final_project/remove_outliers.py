import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import numpy as np
import matplotlib.pyplot as plt

#
# def check_linear_regression():
#     from sklearn.linear_model import LinearRegression
#
#     reg = LinearRegression()
#     reg.fit(feature_train, target_train)
#


def get_slope(feature_train, target_train):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(feature_train, target_train)
    print("slope:", reg.coef_[0])
    return reg.intercept_


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def plot(data, feature_name):
    for point in data:
        plt.scatter(point[0], point[1])
    plt.xlabel("poi")
    plt.ylabel(feature_name)
    plt.savefig(feature_name)
    plt.show()


def start():
    financial_features = ['poi', 'salary', 'total_payments', 'bonus', 'deferred_income', 'total_stock_value',
                          'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock']

    email_features = ['poi', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                      'shared_receipt_with_poi']

    ### Load the dictionary containing the dataset
    data_dict = pickle.load(open("final_project_dataset_unix2.pkl", "rb"))

    ### Task 2: Remove outliers
    data_dict.pop("TOTAL", 0)
    data = featureFormat(data_dict, financial_features, sort_keys=True)

    for index, fin_ft in enumerate(financial_features[1:]):
        plot([ (emp[0], emp[index+1]) for emp in data], financial_features[index+1])




    # labels, features = targetFeatureSplit(data)
    #
    # # data = np.array(financial_outliers)
    # from sklearn.linear_model import LinearRegression
    # reg = LinearRegression()
    #
    # labels, features = targetFeatureSplit(data)


    # reg.fit(features, labels)
    # print("slope:", reg.coef_[0])
    # # return reg.intercept_
    #
    #
    #
    # clean_data = reject_outliers(features, reg.coef_[0])
    # print(clean_data)

start()
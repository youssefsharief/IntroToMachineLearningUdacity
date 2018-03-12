import pickle
from poi_email_addresses import poiEmails

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
    de"""



# men = ["SKILLING JEFFREY K", "LAY KENNETH L", "FASTOW ANDREW S" ]

# x= { item: enron_data[item]["total_payments"] for item in men}

def getNumberOfEmployees():
    enron_data = pickle.load(open("../final_project/final_project_dataset_unix2.pkl", "rb"))
    return len([employee for employee in enron_data])

def getNumberOfEmployeesWithTotalPayments():
    enron_data = pickle.load(open("../final_project/final_project_dataset_unix2.pkl", "rb"))
    return len([employee for employee in enron_data if enron_data[employee]["total_payments"] != 'NaN'])


def getNumberOfPersonsOfInterests():
    enron_data = pickle.load(open("../final_project/final_project_dataset_unix2.pkl", "rb"))
    return len([employee for employee in enron_data if enron_data[employee]["poi"] == 1])

def getNumberOfPersonsOfInterestsInTxt():
    f = open("../final_project/poi_names.txt", "r")
    lines = f.readlines()
    return len(poiEmails())

def getNumberOfFeatures():
    enron_data = pickle.load(open("../final_project/final_project_dataset_unix2.pkl", "rb"))
    return len([employee for employee in enron_data if enron_data[employee]["total_payments"] != 'NaN'])

print(getNumberOfPersonsOfInterestsInTxt())

#!/usr/bin/python3

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

import pickle
import math
enron_data = pickle.load(open("../final_project/final_project_dataset_unix.pkl", "rb"))


# men = ["SKILLING JEFFREY K", "LAY KENNETH L", "FASTOW ANDREW S" ]

# x= { item: enron_data[item]["total_payments"] for item in men}

x = 0
for key in enron_data:
    if enron_data[key]["total_payments"]!='NaN':
        x+=1
print(x)
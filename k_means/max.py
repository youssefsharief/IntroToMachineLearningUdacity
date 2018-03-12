import pickle


def max_and_min(field):
    data_dict = pickle.load(open("../final_project/final_project_dataset_unix2.pkl", "rb"))
    max_salary = 0
    min_salary = float("inf")

    for key in data_dict:
        if float(data_dict[key][field]) > 0 and data_dict[key][field] != "NaN":
            if float(data_dict[key][field]) > max_salary:
                max_salary = data_dict[key][field]
            if float(data_dict[key][field]) < min_salary:
                min_salary = data_dict[key][field]
    print(min_salary, max_salary)


max_and_min("salary")


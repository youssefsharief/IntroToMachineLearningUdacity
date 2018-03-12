from sklearn.preprocessing import MinMaxScaler
import pickle

def rescale(data):
    scaler = MinMaxScaler()
    cl = scaler.get_params(data)

def rescale_salary_and_stock_options():
    data_dict = pickle.load(open("../final_project/final_project_dataset_unix2.pkl", "rb"))
    data = [[data_dict[employee]["salary"] if data_dict[employee]["salary"] != 'NaN' else 0,
             data_dict[employee]["exercised_stock_options"] if data_dict[employee]["exercised_stock_options"]!='NaN' else 0]
            for employee  in data_dict]
    print(data)
    rescale(data)

rescale_salary_and_stock_options()

# trained_model_folder = ""
# pt_intervals = [
#                 [1,3],
#                 [3,5],
#                 [5,8],
#                 [8,12],
#                 [12,24]
# ]

# optuna_file_name = "OptunaStudy"

# for pt_interval in pt_intervals:
    
import pickle

# Path to your pickle file
pickle_file_path = "/home/mdicosta/flowDplus/MLtraining/2050/no_opt_chi2cpa/pt1_3/ModelHandler_pT_1_3.pickle"

# Open and load the pickle file
with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

# Inspect the content
print("Type of data:", type(data))
print("Content of data:", data)

# If it's a dictionary or list, iterate through its elements
if isinstance(data, dict):
    print("Keys:", data.keys())
    print("Values:", data.values())
elif isinstance(data, list):
    for idx, item in enumerate(data):
        print(f"Item {idx}: {item}")
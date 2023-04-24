import pickle

def loading_log_data():
    with open('C:\\auto_forward_transportation\\log_data.pickle', 'rb') as f:
        log_data_load = pickle.load(f)
    return log_data_load

log = loading_log_data()
print(log)

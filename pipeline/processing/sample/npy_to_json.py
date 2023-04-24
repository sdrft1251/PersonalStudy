import os
import json
import numpy as np

from utils import data
from config import config_json


if __name__=="__main__":
    train_rate = config_json["train_data_rate"]
    val_rate = config_json["val_data_rate"]
    processing_list = config_json["processing_list"]
    source_data_path = '../data/source_data'

    ### Data Loading
    normal_datas = []
    for file_name in ['normal_merged_1.npy', 'normal_merged_2.npy']:
        loaded_data = data.load_data(os.path.join(source_data_path, file_name), 256)
        sampled = loaded_data[np.random.permutation(len(loaded_data))[:5]]
        normal_datas.append(sampled)
    abnormal_datas = []
    for file_name in ["abnormal_merged_1.npy", "abnormal_merged_2.npy"]:
        loaded_data = data.load_data(os.path.join(source_data_path, file_name), 256)
        sampled = loaded_data[np.random.permutation(len(loaded_data))[:5]]
        abnormal_datas.append(sampled)

    i = 0
    for datas in normal_datas:
        for da in datas:
            with open(f"../data/test_data/normal/data_{i}.json", "w") as write_file:
                json.dump({'inputs': da.tolist()}, write_file)
            i += 1
 
    
    i = 0
    for datas in abnormal_datas:
        for da in datas:
            with open(f"../data/test_data/abnormal/data_{i}.json", "w") as write_file:
                json.dump({'inputs': da.tolist()}, write_file)
            i += 1
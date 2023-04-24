import os
from scipy.io import arff
import pandas as pd
import numpy as np
import wfdb
import tensorflow as tf

##################################################
##### Data Read
##################################################

# ECG5000 Read
def data_from_ecg5000(folder_path):
    # return Dict
    result = {}

    file_list = os.listdir(folder_path)
    arff_list = [file for file in file_list if file.endswith(".arff")]

    # 데이터 추출 함수
    def return_df_from_arff(arff_data):
        one_data = arff_data[0]
        start = True
        for i in range(140):
            col_name = "att{}".format(i+1)
            one_col = one_data[col_name].astype(float).reshape(-1,1)
            if start:
                tot_data = one_col
                start = False
            else:
                tot_data = np.concatenate((tot_data, one_col), axis=1)
        target_col = one_data["target"].astype(float).reshape(-1,1)
        tot_data = np.concatenate((tot_data, target_col), axis=1)
        return tot_data

    for arf_ in arff_list:
        file_ = arff.loadarff(folder_path+"/"+arf_)
        tot = return_df_from_arff(file_)
        # 확장자 제거 후 추가
        result[arf_[:-5]] = tot
    
    return result

# MIT DataBase
def data_from_mit(folder_path):
    # return Dict
    result = {}

    file_list = os.listdir(folder_path)
    # Record 파일에서 데이터 제목 가져오기
    mit_file_name_list = []
    with open(folder_path+"RECORDS", 'r') as f:
        while True:
            line = f.readline()
            if len(line.strip()) != 0:
                mit_file_name_list.append(line.strip())
            if not line:
                break
    for mi_ in mit_file_name_list:
        signals, fields = wfdb.rdsamp(folder_path+mi_)
        # 두 데이터 추가
        result[mi_] = [signals, fields]
    
    return result

def make_dataformat_from_mit(data_col, name, time_len, over_len):
    signal_1 = data_col[name][0][:,0]
    # Using Only first signal now
    #signal_2 = data_col[name][0][:,1]
    detail = data_col[name][1]
    # Result list
    result = []

    # Slicing Window Start
    start_idx = 0
    while start_idx+time_len <= len(signal_1):
        sample_data = signal_1[start_idx:start_idx+time_len]
        result.append(sample_data)
        start_idx += (time_len-over_len)

    # Make right format
    result = np.array(result).reshape(-1, time_len, 1)
    # Make 0 based data
    min_val = result.min()
    max_val = result.max()
    if min_val != 0:
        result = (result - min_val) / (max_val - min_val)
    else:
        result /= max_val
    return result

# Return Tensor dataset
def tensorset(arr, shape, batch_size, drop_remainder=True):
    # type casting & reshaping
    data = arr.astype(np.float32)
    print("Before reshape : {}".format(data.shape))
    data = np.reshape(data, shape)
    print("After reshape : {} | data type : {}".format(data.shape, data.dtype))
    # make to tensor
    ds = tf.data.Dataset.from_tensor_slices(data).shuffle(buffer_size=data.shape[0]*3)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    return ds


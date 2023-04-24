import os
import numpy as np
import pyedflib
import numpy as np
from scipy.signal import lfilter, sosfilt
from scipy.signal import butter, iirnotch, lfilter
from scipy import signal
import matplotlib.pyplot as plt

##############################################################################################################################
# For EDF
##############################################################################################################################

######################## Find EDF file Path ########################
def EDFDirList(folder_dir):
    file_list = os.listdir(folder_dir)
    edf_path_list = []
    for fold_name in file_list:
        edf_list = os.listdir(folder_dir+"/"+fold_name)
        edf_path_list += [folder_dir+"/"+fold_name+"/"+edf for edf in edf_list if edf.endswith(".edf")]
    print("Total edf file num is : {}".format(len(edf_path_list)))
    return edf_path_list

######################## High & Low pass filter ########################
## A high pass filter allows frequencies higher than a cut-off value
def butter_highpass(cutoff, fs, order=5):
    sos = butter(order, cutoff, 'hp', fs=fs, output='sos')
    return sos

## A low pass filter allows frequencies lower than a cut-off value
def butter_lowpass(cutoff, fs, order=5):
    sos = butter(order, cutoff, 'lp', fs=fs, output='sos')
    return sos

def final_filter(data, fs, order=5):
    highpass_sos = butter_highpass(cutoff_high, fs, order=order)
    x = sosfilt(highpass_sos, data)
    lowpass_sos = butter_highpass(cutoff_high, fs, order=order)
    y = sosfilt(lowpass_sos, x)
    return y

######################## Scaler ########################
class StandardScaler():
    def __init__(self):
        self.mean = 0
        self.std = 1
        
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)
        
    def transform(self, data):
        return (data - self.mean) / self.std

######################## Signal ########################
# EDf 파일에서 데이터 가져오기
def make_signal(edf_dir):
    scaler = StandardScaler()
    with pyedflib.EdfReader(edf_dir) as f:
        ecg = f.readSignal(0)
        ecg = ecg[::2]
        ecg = ecg[ecg>=2]
    # 일정 길이 이상인 데이터만 추출 (데이터 품질을 위해)
    if len(ecg)>= fs*3600:
        ecg = final_filter(ecg, fs, order)
        # 변환 후의 안정성을 위해 일정 길이 만큼 Cut
        ecg = ecg[fs*5:]
        # Scaling
        scaler.fit(ecg)
        ecg = scaler.transform(ecg)
    else:
        ecg = np.array([])
    return ecg

# Sliding window로 데이터 생성
def make_datasample(signal, hz, len_sec, skip_sec):
    # 길이 연산
    time_len = hz*len_sec
    skip_len = hz*skip_sec
    # Window
    result = []
    start_idx = 0
    while start_idx+time_len <= len(signal):
        sample_data = signal[start_idx:start_idx+time_len]
        start_idx += skip_len
        # 길이가 정한 값과 일치하는 값만 Append
        if len(sample_data) == time_len:
            result.append(sample_data)
    # Reshaping
    result = np.array(result).reshape(-1, time_len, 1)
    return result.astype(np.float32)

# Sliding window로 데이터 생성
def data_preprocessing(path_list, hz, len_sec, skip_sec):
    # Append할 numpy array 생성
    data_col = np.zeros((1,hz*len_sec,1), dtype=np.float32)
    # 파일별로 전처리 후 Append
    for idx, edf_path in enumerate(path_list):
        try:
            signal = make_signal(edf_path)
        except:
            signal = np.array([])
        preprocessed_signal = make_datasample(signal=signal, hz=hz, len_sec=len_sec, skip_sec=skip_sec)
        # 빈데이터 Skip
        if len(preprocessed_signal)==0:
            continue
        data_col = np.concatenate((data_col, preprocessed_signal), axis=0)
        if idx%100 == 0:
            print("Now Process Total num : {} | Now num : {} | Data Size : {}".format(len(edf_path_list), idx+1, data_col.shape))

    data_col = data_col[1:]
    return data_col

##############################################################################################################################
# For WFDB
##############################################################################################################################

######################## MIT DataBase에서 Signal 가져오기 ########################
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

######################## 데이터마다 돌면서 전처리 후 Return ########################
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
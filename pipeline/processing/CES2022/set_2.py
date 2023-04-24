import sqlite3
import pyedflib
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.signal import butter, sosfilt
import os
import ast
import pywt

class loading_ex:

    def metadb_connect(self, db_path):
        self.conn = sqlite3.connect(db_path)
    
    def metadb_close(self):
        if self.conn:
            self.conn.close()

    def read_sampled(self, sample_name, window_sec=10, skip_sec=5):
        ##### Variables #####
        one_image_len = 256 * 10   # hz x 1min
        len_per_page = one_image_len * 6
        #####################

        cur = self.conn.cursor()
        cur.execute(f"SELECT id FROM samplegroup WHERE group_name='{sample_name}';")
        results = cur.fetchall()
        if len(results) == 0:
            return -1
        
        group_id = results[0][0]

        cur.execute(f"SELECT ecgtest_id, page FROM samplelink WHERE samplegroup_id={group_id};")
        to_dos = cur.fetchall()
        data_dumps = []

        for (ecgtest_id, page) in to_dos:
            cur.execute(f"SELECT edf_path FROM ecgtest WHERE id={ecgtest_id}")
            results = cur.fetchall()
            if len(results) == 0:
                return -1
            data_path = results[0][0]
            data_path = os.path.join("/home/wellysis-tft/Desktop/code/datalake", data_path[7:])
            f = pyedflib.EdfReader(data_path)
            sigbufs = f.readSignal(chn=0, start=(page-1)*len_per_page, n=len_per_page)
            f._close()

            start = 0
            skip_size = skip_sec*256
            window_size = window_sec*256
            while start+window_size<=len_per_page:
                sliced = sigbufs[start : start+window_size]
                data_dumps.append(sliced)
                start += skip_size
        return np.array(data_dumps)


class loading_smc:
    def get_data(self, path, sampling_rate_to = 256):
        data = np.load(path)
        if sampling_rate_to == 250:
            data_1 = data[:,::2]
            data_2 = data[:,1::2]
            data = (data_1 + data_2)/2
        elif sampling_rate_to == 256:
            sum_idx_0_arr = np.load("/home/wellysis-tft/Desktop/code/datalake/processed_data/CES2022/ref_data/hz256_mean_idx.npy")
            sum_idx_1_arr = sum_idx_0_arr - 1
            idx_arr = np.load("/home/wellysis-tft/Desktop/code/datalake/processed_data/CES2022/ref_data/hz256_idx.npy")
            data[:,sum_idx_0_arr] = (data[:,sum_idx_0_arr] + data[:,sum_idx_1_arr])/2
            data = data[:,idx_arr]
        return data


class loading_simulator:
    def get_data(self, path, window_sec=10, skip_sec=5):
        with open(path, "r") as f:
            datas = f.readlines()
        datas = np.array(ast.literal_eval(datas[0]))
        data_dumps = []
        start = 256*60
        skip_size = skip_sec*256
        window_size = window_sec*256
        while start+window_size<=len(datas):
            sliced = datas[start : start+window_size]
            data_dumps.append(sliced)
            start += skip_size
        return np.array(data_dumps)




################################################################################################
# Data processing
################################################################################################
######################## Reverse Checking ########################
def reversing_data(data):
    max_val = data.max(axis=1)
    min_val = abs(data.min(axis=1))
    reverse_val = np.where(max_val>=min_val, 1, -1).reshape(-1,1)
    data = data*reverse_val.reshape(-1,1)
    return data
######################## High & Low pass filter ########################
## A high pass filter allows frequencies higher than a cut-off value
def butter_highpass(cutoff, fs, order=5):
    sos = butter(order, cutoff, 'hp', fs=fs, output='sos')
    return sos
## A low pass filter allows frequencies lower than a cut-off value
def butter_lowpass(cutoff, fs, order=5):
    sos = butter(order, cutoff, 'lp', fs=fs, output='sos')
    return sos
def final_filter(data, fs, lowest_hz, highest_hz, order=5):
    highpass_sos = butter_highpass(lowest_hz, fs, order=order)
    x = sosfilt(highpass_sos, data, axis=-1)
    lowpass_sos = butter_lowpass(highest_hz, fs, order=order)
    y = sosfilt(lowpass_sos, x, axis=-1)
    return y
def bandwith_process(signal, fs, lowest_hz, highest_hz, order=5):
    processed_signal = final_filter(signal, fs=fs, lowest_hz=lowest_hz, highest_hz=highest_hz, order=order)
    return processed_signal
######################## Normalizing ########################
def normal_scaling(data_arr):
    mean_val = data_arr.mean(axis=-1)
    std_val = data_arr.std(axis=-1)
    # if std == 0 --> error val -> replace to 1
    std_val = np.where(std_val==0, 1, std_val)

    mean_val = mean_val.reshape(-1, 1)
    std_val = std_val.reshape(-1, 1)
    return (data_arr-mean_val)/std_val
######################## FFT ########################
def cwt_processing(signal):
    widths = np.arange(1, 101)
    cwtmatr, freqs = pywt.cwt(data=signal, scales=widths, wavelet='gaus3', axis=-1)
    return abs(np.transpose(cwtmatr, [1,2,0]))
################################################


def merging(data_list, label_list, split_type, split_rates=[0.8, 0.1]):
    train_inputs_dumps = []
    val_inputs_dumps = []
    test_inputs_dumps = []

    train_labels_dumps = []
    val_labels_dumps = []
    test_labels_dumps = []

    for idx, val in enumerate(data_list):
        print(f"Index == {idx} is Start")
        la = label_list[idx]
        label = None
        # Labeling
        if la == "normal":
            label = np.zeros(len(val)).reshape(-1,1)
        else:
            label = np.ones(len(val)).reshape(-1,1)
        
        # Processing
        val = normal_scaling(val)
        val = bandwith_process(val, fs=256, lowest_hz=0.3, highest_hz=50)
        val = cwt_processing(val)

        # Spliting
        sp = split_type[idx]
        if sp == "time":
            train_idx = int(len(val)*split_rates[0])
            val_idx = int(len(val)*(split_rates[0] + split_rates[1]))

            train_inputs = val[:train_idx]
            val_inputs = val[train_idx:val_idx]
            test_inputs = val[val_idx:]

            train_labels = label[:train_idx]
            val_labels = label[train_idx:val_idx]
            test_labels = label[val_idx:]

            print(f"Splited to -----")
            print(f"train_inputs : {train_inputs.shape} | train_labels : {train_labels.shape}")
            print(f"val_inputs : {val_inputs.shape} | val_labels : {val_labels.shape}")
            print(f"test_inputs : {test_inputs.shape} | test_labels : {test_labels.shape}")
        else:   # Random
            first_test_size = 1-split_rates[0]
            secode_test_size = (first_test_size - split_rates[1])/first_test_size
            train_inputs, remain_inputs, train_labels, remain_label = train_test_split(val, label, test_size=1-split_rates[0], random_state=42)
            val_inputs, test_inputs, val_labels, test_labels = train_test_split(remain_inputs, remain_label, test_size=secode_test_size, random_state=42)
            print(f"Splited to -----")
            print(f"train_inputs : {train_inputs.shape} | train_labels : {train_labels.shape}")
            print(f"val_inputs : {val_inputs.shape} | val_labels : {val_labels.shape}")
            print(f"test_inputs : {test_inputs.shape} | test_labels : {test_labels.shape}")

        train_inputs_dumps.append(train_inputs)
        val_inputs_dumps.append(val_inputs)
        test_inputs_dumps.append(test_inputs)

        train_labels_dumps.append(train_labels)
        val_labels_dumps.append(val_labels)
        test_labels_dumps.append(test_labels)

    return (np.concatenate(train_inputs_dumps, axis=0), np.concatenate(train_labels_dumps, axis=0)),\
    (np.concatenate(val_inputs_dumps, axis=0), np.concatenate(val_labels_dumps, axis=0)),\
    (np.concatenate(test_inputs_dumps, axis=0), np.concatenate(test_labels_dumps, axis=0))

    


        


if __name__ == "__main__":
    ex = loading_ex()
    ex.metadb_connect("/home/wellysis-tft/Desktop/code/datalake/data/ecg_meta.db")
    ex_normal_1 = ex.read_sampled(sample_name="CES_2022_UoA_Normal", window_sec=10, skip_sec=1)
    ex_normal_2 = ex.read_sampled(sample_name="CES_2022_Liverpool_Normal", window_sec=10, skip_sec=1)
    ex.metadb_close()
    
    smc = loading_smc()
    smc_normals_1 = smc.get_data("/home/wellysis-tft/Desktop/code/CES2022/data/source_data/normal_merged_1.npy")
    smc_normals_2 = smc.get_data("/home/wellysis-tft/Desktop/code/CES2022/data/source_data/normal_merged_2.npy")
    smc_abnormals_1 = smc.get_data("/home/wellysis-tft/Desktop/code/CES2022/data/source_data/abnormal_merged_1.npy")
    smc_abnormals_2 = smc.get_data("/home/wellysis-tft/Desktop/code/CES2022/data/source_data/abnormal_merged_2.npy")

    simulator = loading_simulator()
    simulator_normal_1 = simulator.get_data("/home/wellysis-tft/Desktop/code/datalake/data/simulator/40bpm.txt", window_sec=10, skip_sec=1)
    simulator_normal_2 = simulator.get_data("/home/wellysis-tft/Desktop/code/datalake/data/simulator/80bpm.txt", window_sec=10, skip_sec=1)
    simulator_normal_3 = simulator.get_data("/home/wellysis-tft/Desktop/code/datalake/data/simulator/120bpm.txt", window_sec=10, skip_sec=1)
    
    data_list = [ex_normal_1, ex_normal_2, smc_normals_1, smc_normals_2, smc_abnormals_1, smc_abnormals_2, simulator_normal_1, simulator_normal_2, simulator_normal_3]
    label_list = ["normal", "normal", "normal", "normal", "abnormal", "abnormal", "normal", "normal", "normal"]
    split_type = ["random", "random", "time", "time", "time", "time", "random", "random", "random"]
    split_rates = [0.8, 0.1]
    (a,b),(c,d),(e,f) = merging(data_list, label_list, split_type, split_rates)
    print(a.shape, b.shape, c.shape, d.shape, e.shape, f.shape)


    base_path = "/home/wellysis-tft/Desktop/code/datalake/processed_data/CES2022/set_3"

    np.save(os.path.join(base_path, "train_inputs"), a)
    np.save(os.path.join(base_path, "train_labels"), b)
    np.save(os.path.join(base_path, "val_inputs"), c)
    np.save(os.path.join(base_path, "val_labels"), d)
    np.save(os.path.join(base_path, "test_inputs"), e)
    np.save(os.path.join(base_path, "test_labels"), f)


    
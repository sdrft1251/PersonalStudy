import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
file_list = os.listdir("./data")
print(len(file_list))
print(file_list)

train_feature = pd.read_csv("./data/"+file_list[2])
train_label = pd.read_csv("./data/"+file_list[3])
sub_df = pd.read_csv("./data/"+file_list[0])
sub_feature = pd.read_csv("./data/"+file_list[1])
print(train_feature.shape, train_label.shape)


def return_index(arr, num):
    return arr.argsort()[-num:]

def prerpocessing_using_static(features_set, label_set):
    input_set = []
    target_set = []
    ambiguous           =   [0,32,50]
    static              =   [8,9,12,22,26,28,29,30,31,38,39,48,49,52]
    Rotary_motion       =   [18,27,35,36,40,41,51,53,54,55,56,57]
    vertical_motion     =   [6,10,11,13,14,16,17,23,24,33,37,42,43,44,45,46,47,60,58,59]
    Rotary_vertical     =   [1,2,3,4,5,7,15,19,20,21,25,34]
    
    for id_idx in range(features_set.id.min(), features_set.id.max()+1):
        sample_features = features_set[["acc_x","acc_y","acc_z","gy_x","gy_y","gy_z"]][features_set.id==id_idx].values
        
        #Delete noise & principa component
        sample_features_cum = []
        for idx in range(1, sample_features.shape[0]+1):
            sample_features_cum.append(sample_features[:idx,:].sum(axis=0))
        sample_features_cum = np.array(sample_features_cum)
        principal_eigen_vector = (sample_features_cum[-1,:] - sample_features_cum[0,:]) / len(sample_features_cum)
        
        sample_features_cum_kl = []
        for idx in range(1, sample_features_cum.shape[0]):
            sample_features_cum_kl.append(sample_features_cum[idx,:] - principal_eigen_vector*idx - sample_features_cum[idx])
        sample_features_cum_kl = np.array(sample_features_cum_kl)
        
        sample_features_cum_kl_max = abs(sample_features_cum_kl).max(axis=0)
        sample_features_cum_kl_max = np.where(sample_features_cum_kl_max==0,1,sample_features_cum_kl_max)
        sample_features_cum_kl_norm = sample_features_cum_kl / sample_features_cum_kl_max
        
        #Fourier transform
        fmax = 50
        dt = 1/fmax
        N  = 600
        t  = np.arange(0,N)*dt
        df = fmax/N
        f = np.arange(0,N)*df
        xf = np.fft.fft(sample_features_cum_kl_norm, axis=0)*dt
        xf[:,0]
        tq_index=f[0:int(N/2+1)]
        tq_abs= np.abs(xf[0:int(N/2+1)])
        
        freq_x = tq_index[return_index(tq_abs[:,0],5)].mean()
        freq_y = tq_index[return_index(tq_abs[:,1],5)].mean()
        freq_z = tq_index[return_index(tq_abs[:,2],5)].mean()
        freq_gy_x = tq_index[return_index(tq_abs[:,3],5)].mean()
        freq_gy_y = tq_index[return_index(tq_abs[:,4],5)].mean()
        freq_gy_z = tq_index[return_index(tq_abs[:,5],5)].mean()
        
        
        #Mean values
        mean_arr = sample_features_cum_kl_norm.mean(axis=0)
        #Mag values
        mag_arr = abs(sample_features_cum_kl_norm).mean(axis=0)
        #Std values
        std_arr = sample_features_cum_kl_norm.std(axis=0)
        #Cov values
        cov_xy = np.cov(sample_features_cum_kl_norm[:,0], sample_features_cum_kl_norm[:,1])[0][1]
        cov_yz = np.cov(sample_features_cum_kl_norm[:,1], sample_features_cum_kl_norm[:,2])[0][1]
        cov_zx = np.cov(sample_features_cum_kl_norm[:,2], sample_features_cum_kl_norm[:,0])[0][1]
        #Corr values
        cor_xy = cov_xy / (std_arr[0]*std_arr[1])
        cor_yz = cov_yz / (std_arr[1]*std_arr[2])
        cor_zx = cov_zx / (std_arr[2]*std_arr[0])
        
        #Energy values
        shift_0_values = np.roll(sample_features_cum_kl_norm, 1, axis=0)[1:]
        shift_1_values = sample_features_cum_kl_norm[1:]
        energys = np.power((shift_0_values-shift_1_values),2)
        energy_mean = energys.mean(axis=0)
        energy_std = energys.std(axis=0)
        
        id_sample_set = [freq_x, freq_y, freq_z, freq_gy_x, freq_gy_y, freq_gy_z] + list(mean_arr)\
        + list(mag_arr) + list(std_arr) + [cor_xy, cor_yz, cor_zx] + list(energy_mean) + list(energy_std)
        target_label = label_set[label_set.id == id_idx].label.values[0]
        target_label_idx = -1
        if target_label in ambiguous:
            target_label_idx = 0
        elif target_label in static:
            target_label_idx = 1
        elif target_label in Rotary_motion:
            target_label_idx = 2
        elif target_label in vertical_motion:
            target_label_idx = 3
        elif target_label in Rotary_vertical:
            target_label_idx = 4
        
        #Append
        input_set.append(id_sample_set)
        target_set.append(target_label_idx)
    
    input_set = np.array(input_set)
    target_set = np.array(target_set)
    print("Input set : {} / Target set : {}".format(input_set.shape, target_set.shape))
    return input_set, target_set

 columns_list=["freq_x", "freq_y", "freq_z",
               "freq_gy_x", "freq_gy_y", "freq_gy_z",
               "mean_x", "mean_y", "mean_z",
               "mean_gy_x", "mean_gy_y", "mean_gy_z",
               "mag_x", "mag_y", "mag_z",
               "mag_gy_x", "mag_gy_y", "mag_gy_z",
               "std_x", "std_y", "std_z",
               "std_gy_x", "std_gy_y", "std_gy_z",
               "cor_xy", "cor_yz", "cor_zx",
               "energy_mean_x", "energy_mean_y", "energy_mean_z",
               "energy_mean_gy_x", "energy_mean_gy_y", "energy_mean_gy_z",
               "energy_std_x", "energy_std_y", "energy_std_z",
               "energy_std_gy_x", "energy_std_gy_y", "energy_std_gy_z"]

etas = []
for feature_idx in range(input_set.shape[1]):
    measur_arr = input_set[:,feature_idx]
    cat_arr =  target_set
    eta = correlation_ratio(categories=target_set, measurements=measur_arr)
    etas.append(eta)
    
df = pd.DataFrame(np.array(etas).reshape(1,-1), columns=columns_list)




def return_index(arr, num):
    return np.flip(arr.argsort()[-num:])

def prerpocessing_using_static(features_set, label_set):
    input_set = []
    target_set = []
    ambiguous           =   [0,32,50]
    static              =   [8,9,12,22,26,28,29,30,31,38,39,48,49,52]
    Rotary_motion       =   [18,27,35,36,40,41,51,53,54,55,56,57]
    vertical_motion     =   [6,10,11,13,14,16,17,23,24,33,37,42,43,44,45,46,47,60,58,59]
    Rotary_vertical     =   [1,2,3,4,5,7,15,19,20,21,25,34]
    
    for id_idx in range(features_set.id.min(), features_set.id.max()+1):
        sample_features = features_set[["acc_x","acc_y","acc_z","gy_x","gy_y","gy_z"]][features_set.id==id_idx].values
        
        #Delete noise & principa component
        sample_features_cum = []
        for idx in range(1, sample_features.shape[0]+1):
            sample_features_cum.append(sample_features[:idx,:].sum(axis=0))
        sample_features_cum = np.array(sample_features_cum)
        principal_eigen_vector = (sample_features_cum[-1,:] - sample_features_cum[0,:]) / len(sample_features_cum)
        
        sample_features_cum_kl = []
        for idx in range(0, sample_features_cum.shape[0]):
            sample_features_cum_kl.append(sample_features_cum[idx,:] - principal_eigen_vector*idx - sample_features_cum[0,:])
        sample_features_cum_kl = np.array(sample_features_cum_kl)
        
        sample_features_cum_kl_max = abs(sample_features_cum_kl).max(axis=0)
        sample_features_cum_kl_max = np.where(sample_features_cum_kl_max==0,1,sample_features_cum_kl_max)
        sample_features_cum_kl_norm = sample_features_cum_kl / sample_features_cum_kl_max
        
        fmax = 50
        dt = 1/fmax
        N  = 600
        t  = np.arange(0,N)*dt
        df = fmax/N
        f = np.arange(0,N)*df
        xf = np.fft.fft(sample_features_cum_kl_norm, axis=0)*dt
        tq_index=f[0:int(N/2+1)]
        tq_abs= np.abs(xf[0:int(N/2+1)])
        
        target_label = label_set[label_set.id == id_idx].label.values[0]
        target_label_idx = -1
        if target_label in ambiguous:
            target_label_idx = 0
        elif target_label in static:
            target_label_idx = 1
        elif target_label in Rotary_motion:
            target_label_idx = 2
        elif target_label in vertical_motion:
            target_label_idx = 3
        elif target_label in Rotary_vertical:
            target_label_idx = 4
        
        #Append
        input_set.append(tq_abs)
        target_set.append(target_label)
    
    input_set = np.array(input_set)
    target_set = np.array(target_set)
    print("Input set : {} / Target set : {}".format(input_set.shape, target_set.shape))
    return input_set, target_set

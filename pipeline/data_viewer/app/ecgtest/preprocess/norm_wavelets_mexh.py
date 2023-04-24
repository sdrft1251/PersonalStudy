import pywt
import numpy as np

def return_cwt_with_maxc(signal):
    widths = np.arange(1, 101)
    cwtmatr, freqs = pywt.cwt(data=signal, scales=widths, wavelet='mexh', axis=-1)
    argmax_idx = np.argmax(cwtmatr.max(axis=-1), axis=0)
    return cwtmatr[argmax_idx]


def normal_scaling(data_arr):
    mean_val = data_arr.mean(axis=-1)
    std_val = data_arr.std(axis=-1)
    # if std == 0 --> error val -> replace to 1
    std_val = np.where(std_val==0, 1, std_val)

    return (data_arr-mean_val)/std_val


def processing(signal):
    signal = normal_scaling(signal)
    return return_cwt_with_maxc(signal)
import pywt
import numpy as np
from scipy.signal import argrelextrema
from scipy.signal import find_peaks

def return_cwt_with_mexh(signal):
    widths = np.arange(1, 101)
    cwtmatr, freqs = pywt.cwt(data=signal, scales=widths, wavelet='gaus3', axis=-1)
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
    wavelets_signal = return_cwt_with_mexh(signal)
    # print(wavelets_signal)

    # local_maxima_idx = argrelextrema(wavelets_signal, np.greater)
    local_maxima_idx, _ = find_peaks(wavelets_signal, distance=30)
    local_maxima = wavelets_signal[local_maxima_idx]
    print(local_maxima.shape)
    # print(local_maxima_idx)

    hist, bin_edges = np.histogram(local_maxima, bins=20)
    print(hist)
    print(bin_edges)
    h = (hist*bin_edges[:-1]).sum() / bin_edges[:-1].sum()
    print(h)
    mp = np.where(wavelets_signal>=h, 1, 0)

    signal_qrs = signal*mp
    return signal_qrs



import pywt
import numpy as np
from scipy.signal import argrelextrema

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

    local_maxima_idx = argrelextrema(wavelets_signal, np.greater)
    local_maxima = wavelets_signal[local_maxima_idx]

    hist, bin_edges = np.histogram(local_maxima, bins=50)
    h = (hist*bin_edges[:-1]).sum() / bin_edges[:-1].sum()
    mp = np.where(wavelets_signal>=h, 1, 0)

    signal_qrs = signal*mp
    signal_qrs_else = signal-signal_qrs

    wavelets_signal_2 = return_cwt_with_mexh(signal_qrs_else)
    local_maxima_idx_2 = argrelextrema(wavelets_signal_2, np.greater)
    local_maxima_2 = wavelets_signal_2[local_maxima_idx_2]

    hist_2, bin_edges_2 = np.histogram(local_maxima_2, bins=50)
    h_2 = (hist_2*bin_edges_2[:-1]).sum() / bin_edges_2[:-1].sum()
    mp_2 = np.where(wavelets_signal_2>=h_2, 1, 0)

    signal_t = signal_qrs_else*mp_2
    signal_p = signal_qrs_else-signal_t
    return signal_p



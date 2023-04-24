import pywt
import numpy as np
from scipy.signal import butter, sosfilt
import numpy as np


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


def return_cwt_with_mexh(signal):
    widths = np.arange(1, 101)
    cwtmatr, freqs = pywt.cwt(data=signal, scales=widths, wavelet='gaus1', axis=-1)
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
    signal = final_filter(signal, 256, 0.5, 30)
    return return_cwt_with_mexh(signal)



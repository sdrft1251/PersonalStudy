from scipy.signal import butter, sosfilt



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



def processing(signal):
    return final_filter(signal, fs=256, lowest_hz=0.5, highest_hz=30, order=5)

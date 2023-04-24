import os
import pyedflib
from io import BytesIO
import matplotlib.pyplot as plt
from scipy.signal import lfilter, sosfilt
from scipy.signal import butter, iirnotch, lfilter
from viewer.config import Config

from viewer import app
from flask import send_file

def make_tree(path):
    tree = dict(name=path, children=[])
    try: lst = os.listdir(path)
    except OSError:
        pass #ignore errors
    else:
        for name in lst:
            fn = os.path.join(path, name)
            if os.path.isdir(fn):
                tree['children'].append(make_tree(fn))
            else:
                tree['children'].append(dict(name=fn))
    return tree

@app.route("/")
def home():
    tree = make_tree(Config.ECG_FILE_PATH)
    return tree

def data_load(path):
    f = pyedflib.EdfReader(path)
    sigbufs = f.readSignal(0)
    f._close()
    print(sigbufs)
    return sigbufs

@app.route("/test")
def read_data():
    edf_path = Config.TEST_FILE_PATH
    print(edf_path)
    f = pyedflib.EdfReader(edf_path)
    sigbufs = f.readSignal(0)
    f._close()
    print(sigbufs)
    return "success"

def ploting_data(signal):
    fig = plt.figure(figsize=(15, 2))
    plt.title("Test ECG")
    plt.plot(signal, 'k--', label='ecg')

    img = BytesIO()
    plt.savefig(img, format='png', dpi=200)
    img.seek(0)
    return img

@app.route("/imagetest")
def plot_data():
    edf_path = Config.TEST_FILE_PATH
    sigbufs = data_load(edf_path)
    sigbufs = sigbufs[1000:2000]

    fig = plt.figure(figsize=(15, 2))
    plt.title("Test ECG")
    plt.plot(sigbufs, 'k--', label='ecg')
    img = BytesIO()
    plt.savefig(img, format='png', dpi=200)
    img.seek(0)
    return send_file(img, mimetype='image/png')


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
    highpass_sos = butter_highpass(0.5, fs, order=order)
    x = sosfilt(highpass_sos, data)
    lowpass_sos = butter_highpass(2.5, fs, order=order)
    y = sosfilt(lowpass_sos, x)
    return y


@app.route("/preprocessingtest")
def preprocess_data():
    edf_path = Config.TEST_FILE_PATH
    sigbufs = data_load(edf_path)
    ecg = final_filter(sigbufs, fs=256, order=5)
    ecg = ecg[1000:2000]
    img = ploting_data(ecg)
    return send_file(img, mimetype='image/png')
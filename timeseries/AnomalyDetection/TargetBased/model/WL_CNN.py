import tensorflow as tf
from scipy.signal import butter, sosfilt
import pywt

class NormalizeLayer(tf.keras.layers.Layer):
    def __init__(self, name="NormalizeLayer"):
        super(NormalizeLayer, self).__init__(name=name)ß

    def call(self, inputs):
        inputs_mean = tf.reshape(tf.math.reduce_mean(inputs, axis=-1), [-1,1])
        inputs_std = tf.reshape(tf.math.reduce_std(inputs, axis=-1), [-1,1])
        return tf.math.divide(tf.math.subtract(inputs, inputs_mean), inputs_std)


class ButterWorthLayer(tf.keras.layers.Layer):
    def __init__(self, lowest_hz, highest_hz, fs=256, order=5, name="ButterWorthLayer"):
        super(ButterWorthLayer, self).__init__(name=name)
        self.lowest_hz = lowest_hz
        self.highest_hz = highest_hz
        self.fs = fs
        self.order = order

     ## A high pass filter allows frequencies higher than a cut-off value
    def butter_highpass(self):
        sos = butter(self.order, self.lowest_hz, 'hp', fs=self.fs, output='sos')
        return sos
    ## A low pass filter allows frequencies lower than a cut-off value
    def butter_lowpass(self):
        sos = butter(self.order, self.highest_hz, 'lp', fs=self.fs, output='sos')
        return sos

    def final_filter(self, data):
        highpass_sos = self.butter_highpass()
        x = sosfilt(highpass_sos, data, axis=-1)
        lowpass_sos = self.butter_lowpass()
        y = sosfilt(lowpass_sos, x, axis=-1)
        return y

    def call(self, inputs):
        return self.final_filter(inputs)


class WaveletsTransLayer(tf.keras.layers.Layer):
    def __init__(self, wavelet='gaus3', scales=np.arange(1,101), name="WaveletsTransLayer"):
        super(WaveletsTransLayer, self).__init__(name=name)
        self.scales = scales
        self.wavelet = wavelet

    def call(self, inputs):
        cwtmatr, freqs = pywt.cwt(data=inputs, scales=self.scales, wavelet=self.wavelet, axis=-1)
        return cwtmatr

def wl_cnn(input_shape=(2560), wavelet='gaus3', scales=np.arange(1,101), name="WL_DNN"):
    inputs = tf.keras.Input(shape=input_shape, name="inputs")
    x = NormalizeLayer()(inputs)
    x = ButterWorthLayer(lowest_hz=lowest_hz, highest_hz=highest_hz)(x)
    x = WaveletsTransLayer(wavelet=wavelet, scales=scales)(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=10)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=10)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=5)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=5)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=5)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(2, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name=name)
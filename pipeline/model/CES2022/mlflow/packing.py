import os
import json
import mlflow
import numpy as np
import mlflow.pyfunc
import cloudpickle
from sys import version_info
PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)


import mlflow.pyfunc
class ModelWrapper(mlflow.pyfunc.PythonModel):

    def preprocessing_module(self, signal):
        import numpy as np
        from scipy.signal import sosfilt, butter

        sampling_rate = 256
        lowest_hz = 0.3
        highest_hz = 50

        # Norm Scaling
        mean_val = signal.mean(axis=-1)
        std_val = signal.std(axis=-1)
        std_val = np.where(std_val==0, 1, std_val)   # if std == 0 --> error val -> replace to 1
        mean_val = mean_val.reshape(-1, 1)
        std_val = std_val.reshape(-1, 1)
        signal =  (signal-mean_val)/std_val

        # Band-with processing
        def butter_lowpass():
            sos = butter(5, highest_hz, 'lp', fs=sampling_rate, output='sos')
            return sos
        def butter_highpass():
            sos = butter(5, lowest_hz, 'hp', fs=sampling_rate, output='sos')
            return sos
        def final_filter(data):
            highpass_sos = butter_highpass()
            x = sosfilt(highpass_sos, data)
            lowpass_sos = butter_lowpass()
            y = sosfilt(lowpass_sos, x)
            return y
        signal = final_filter(signal)

        # FFT
        fft_trans = abs(np.fft.fft(signal, norm="ortho", axis=-1))
        index = np.fft.fftfreq((sampling_rate)*10, 1/(sampling_rate))
        signal = fft_trans[:,np.logical_and(index>=lowest_hz, index<=highest_hz)]

        return signal


    def load_context(self, context):
        self.keras_model = mlflow.keras.load_model(context.artifacts["keras_model"])

    def predict(self, context, model_input):
        data_arr = np.array(model_input).reshape(-1,2560)
        processed_data = self.preprocessing_module(data_arr)
        prediction = self.keras_model.predict(processed_data)
        result = { 'normal': prediction[0][0], 'abnormal': prediction[0][1], 'result': 'normal' if prediction[0][0] > 0.25 else 'abnormal'}
        return result


exp_id = "4"
run_id = "7735749d6d08453fad9263226782a42e"

artifacts = {
        "keras_model": os.path.join("/home/wellysis-tft/Desktop/code/CES2022/mlflow/mlruns/", exp_id, run_id, "artifacts/model")
    }

conda_env = {
    'channels': ['defaults'],
    'dependencies': [
    'python={}'.format(PYTHON_VERSION),
    'pip',
    {
        'pip': [
        'mlflow',
        'keras==2.7.0',
        'pandas==1.3.5',
        'pillow==8.4.0',
        'scipy==1.7.3',
        'tensorflow==2.7.0',
        'numpy==1.21.4',
        'cloudpickle=={}'.format(cloudpickle.__version__),
        ],
    },
    ],
    'name': f'{exp_id}_{run_id}_env'
}


mlflow_pyfunc_model_path = f"/home/wellysis-tft/Desktop/code/CES2022/mlflow/serving_model/{exp_id}/{run_id}"
mlflow.pyfunc.save_model(
    path=mlflow_pyfunc_model_path, python_model=ModelWrapper(), artifacts=artifacts,
    conda_env=conda_env)

print("Save End!!!")
loaded_model = mlflow.pyfunc.load_model(os.path.join(mlflow_pyfunc_model_path))

def get_file_list(path):
    return [file for file in os.listdir(path) if file.endswith(".json")]

def get_file(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

noraml_path = "/home/wellysis-tft/Desktop/code/CES2022/data/test_data/normal"
abnoraml_path = "/home/wellysis-tft/Desktop/code/CES2022/data/test_data/abnormal"
normal_list = get_file_list(noraml_path)
abnormal_list = get_file_list(abnoraml_path)

print("Normal data Test ----")
for file_ in normal_list:
    inputs = get_file(os.path.join(noraml_path, file_))["inputs"]
    print(type(inputs))
    print(len(inputs))
    test_predictions = loaded_model.predict(inputs)
    print(test_predictions)

print("Abnormal data Test ----")
for file_ in abnormal_list:
    inputs = get_file(os.path.join(abnoraml_path, file_))["inputs"]
    print(type(inputs))
    print(len(inputs))
    test_predictions = loaded_model.predict(inputs)
    print(test_predictions)



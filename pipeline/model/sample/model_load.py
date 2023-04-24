import os
import numpy as np
import mlflow
from config import config_json
from sklearn.metrics import accuracy_score


def print_acc(model, inputs, labels):
    predict = model.predict(inputs)
    pred_idx_list=[]
    for pred in predict:
        pred_idx_list.append(np.argmax(pred))
    pred_idx_arr = np.array(pred_idx_list, dtype=np.float32)
    test_accuracy_score = accuracy_score(labels, pred_idx_arr)
    return test_accuracy_score


if __name__=="__main__":


    ################################################################################ Data Pipeline
    from utils import data, ploting
    train_rate = config_json["train_data_rate"]
    val_rate = config_json["val_data_rate"]
    processing_list = config_json["processing_list"]
    source_data_path = config_json["source_data_path"]
    ### Data Loading
    normal_datas = []
    for file_name in config_json["normal_file_names"]:
        loaded_data = data.load_data(os.path.join(source_data_path, file_name))
        normal_datas.append(loaded_data)
    abnormal_datas = []
    for file_name in config_json["abnormal_file_names"]:
        loaded_data = data.load_data(os.path.join(source_data_path, file_name))
        abnormal_datas.append(loaded_data)
    ### Data processing
    (train_inputs, train_labels), (val_inputs, val_labels), (test_inputs, test_labels)\
        = data.pipeline(normal_datas=normal_datas, abnormal_datas=abnormal_datas, train_rate=train_rate, val_rate=val_rate, processing_list=processing_list)
    
    exp_id = "3"
    runi_id = "e9cc34d5523b4f858187425ed9313da1"

    ### Model Loading
    keras_model = mlflow.keras.load_model(os.path.join("../mlflow/mlruns/", exp_id, runi_id, "artifacts/model"))
    print(keras_model.summary())


    print(f"Accuracy Score is Set 1 : {print_acc(keras_model, train_inputs, train_labels)}")
    print(f"Accuracy Score is Set 2 : {print_acc(keras_model, val_inputs, val_labels)}")
    print(f"Accuracy Score is Set 3 : {print_acc(keras_model, test_inputs, test_labels)}")

    save_path = os.path.join("../model_h5_dumps", exp_id, runi_id)
    os.makedirs(save_path, exist_ok=True)
    keras_model.save(os.path.join(save_path, "saved_model_e99.h5"))
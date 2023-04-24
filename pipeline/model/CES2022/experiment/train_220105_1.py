import os
import numpy as np
import mlflow
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd

def plot_cm(y_true, y_pred, col=["Normal", "Abnormal"], figsize=(10,10)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    col = col
    cm = pd.DataFrame(cm, index=col, columns=col)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
    return fig

def plot_image(arr, figsize=(15,3)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(arr)
    return fig

config = {
    "data_source": "/works/datalake/processed_data/CES2022/set_2"
}

if __name__=="__main__":
    ################################################################################ MLFLOW Opt
    mlflow_artifact_save_path = "file:///works/CES2022/mlflow/mlruns"
    mlflow.set_tracking_uri(mlflow_artifact_save_path)
    mlflow.set_experiment("CES2022")
    mlflow.tensorflow.autolog()

    ################################################################################ GET Data
    train_inputs = np.load(os.path.join(config["data_source"], "train_inputs.npy"))
    val_inputs = np.load(os.path.join(config["data_source"], "val_inputs.npy"))
    train_labels = np.load(os.path.join(config["data_source"], "train_labels.npy"))
    val_labels = np.load(os.path.join(config["data_source"], "val_labels.npy"))
    test_inputs = np.load(os.path.join(config["data_source"], "test_inputs.npy"))
    test_labels = np.load(os.path.join(config["data_source"], "test_labels.npy"))
    print(f"train_inputs : {train_inputs.shape} | train_labels : {train_labels.shape}")
    print(f"val_inputs : {val_inputs.shape} | val_labels : {val_labels.shape}")
    print(f"test_inputs : {test_inputs.shape} | test_labels : {test_labels.shape}")


    ######################### Model reconstruct #########################
    exp_id = "4"
    runi_id = "0e0ae201e9b34821b585dd01c24cb193"
    ### Model Loading
    model_1 = mlflow.keras.load_model(os.path.join("/works/CES2022/mlflow/mlruns", exp_id, runi_id, "artifacts/model"))
    print(model_1.summary())


    def return_model():
        input_tens = tf.keras.Input(shape=(498), name="fin_input")
        x = model_1.layers[1](input_tens)
        for lay in model_1.layers[2:]:
            x = lay(x)
        model = tf.keras.Model(inputs=input_tens, outputs=x, name="fin_2")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
        print("Final model")
        print(model.summary())
        return model

    fin_model = return_model()

    with mlflow.start_run(run_name="FFT_DNN_220105_1", tags={"sub": "fin", "v": "0.2.2", "data_source": config["data_source"], "describe": "pre1 + pre2 + Expanding layer"}) as run:
        fin_model.fit(train_inputs, train_labels, epochs=1000, batch_size=128, validation_data=(val_inputs,val_labels))
        print("Logged data and model in run: {}".format(run.info.run_id))

        test_predict = fin_model.predict(test_inputs)
        pred_idx_list=[]
        for pred in test_predict:
            pred_idx_list.append(np.argmax(pred))
        pred_idx_arr = np.array(pred_idx_list, dtype=np.float32)

        test_accuracy_score = accuracy_score(test_labels, pred_idx_arr)
        fig = plot_cm(test_labels, pred_idx_arr)
        fig.savefig("./tmp/Confusion_Mat.png")
        plt.close(fig)
        mlflow.log_metric("Test_ACC", test_accuracy_score)
        mlflow.log_artifact("./tmp/Confusion_Mat.png")

        ################################################################################ Sample save
        train_random_sample_idx = np.random.permutation(len(train_labels))[0]
        val_random_sample_idx = np.random.permutation(len(val_labels))[0]
        test_random_sample_idx = np.random.permutation(len(test_labels))[0]

        train_random_sample = train_inputs[train_random_sample_idx]
        val_random_sample = val_inputs[val_random_sample_idx]
        test_random_sample = test_inputs[test_random_sample_idx]

        train_random_fig = plot_image(train_random_sample)
        val_random_fig = plot_image(val_random_sample)
        test_random_fig = plot_image(test_random_sample)
        train_random_fig.savefig("./tmp/train_random_fig.png")
        val_random_fig.savefig("./tmp/val_random_fig.png")
        test_random_fig.savefig("./tmp/test_random_fig.png")
        mlflow.log_artifact("./tmp/train_random_fig.png")
        mlflow.log_artifact("./tmp/val_random_fig.png")
        mlflow.log_artifact("./tmp/test_random_fig.png")

        ################################################################################ Additional record for saving
        mlflow.set_tag("run_id", run.info.run_id)

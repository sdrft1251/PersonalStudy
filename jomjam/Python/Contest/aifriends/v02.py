import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import deque

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

train_df = pd.read_csv("./data/train.csv")
train_df.drop(columns=["X14","X16","X19"], inplace=True)
print(train_df.shape)

x_parts = np.array(train_df[train_df.columns[1:38]].iloc[:-432], dtype=np.float32)
x_18parts = np.array(train_df[train_df.columns[1:38]].iloc[-432-5:], dtype=np.float32)
y_parts = np.array(train_df[train_df.columns[38:-1]].iloc[:-432], dtype=np.float32)
y_parts_mean = y_parts.mean(axis=1).reshape(-1,1)
y_parts_max = y_parts.max(axis=1).reshape(-1,1)
y_parts_min = y_parts.min(axis=1).reshape(-1,1)
y_parts = np.concatenate((y_parts_max, y_parts_min, y_parts_mean), axis=1)
#y_parts = np.array(train_df["Y10"].iloc[:-432], dtype=np.float32).reshape(-1)
y_18parts = np.array(train_df[train_df.columns[-1:]].iloc[-432:], dtype=np.float32).reshape(-1)
print(x_parts.shape, x_18parts.shape, y_parts.shape, y_18parts.shape)

x_parts_2=[]
for idx in range(len(x_parts)):
    if idx==0:
        x_parts_2.append(np.zeros(37))
    else:
        diff_raw=x_parts[idx]-x_parts[idx-1]
        x_parts_2.append(diff_raw)
x_parts_2 = np.array(x_parts_2, dtype=np.float32)

x_parts = np.concatenate((x_parts, x_parts_2), axis=1)

temp = x_parts[:-144*5]
temp_mean = temp.mean(axis=0)
temp_std = temp.std(axis=0)

dq = deque(maxlen=6)
x=[]
y=[]
for idx in range(len(x_parts)):
    dq.append(x_parts[idx])
    if len(dq)==6:
        temp_ = np.array(dq, dtype=np.float32)
        temp_ = (temp_-temp_mean)/temp_std
        x.append(temp_)
        y.append(np.array(y_parts[idx], dtype=np.float32))
x = np.array(x)
y = np.array(y)
print(x.shape, y.shape)

x_train = x[:-144*5]
x_test = x[-144*5:]
y_train = y[:-144*5]
y_test = y[-144*5:]

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train = x_train.reshape(-1,6,74,1)
x_test = x_test.reshape(-1,6,74,1)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


def return_model():
    input_tens = tf.keras.Input(shape=(6,74,1))
    x = tf.keras.layers.Permute((2,1,3))(input_tens)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(64, kernel_size=6, padding="same"))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.ReLU())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(64, kernel_size=6, padding="same"))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.ReLU())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(64, kernel_size=6, padding="same"))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.ReLU())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(64, kernel_size=6, padding="same"))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.ReLU())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(64, kernel_size=6, padding="same"))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.ReLU())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool1D(pool_size=2))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(64, kernel_size=3, padding="same"))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.ReLU())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(64, kernel_size=3, padding="same"))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.ReLU())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(64, kernel_size=3, padding="same"))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.ReLU())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(64, kernel_size=3, padding="same"))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.ReLU())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(64, kernel_size=3, padding="same"))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.ReLU())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(3, kernel_size=3, padding="same"))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.ReLU())(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inputs=input_tens, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError(), metrics=["mae"])
    print(model.summary())
    return model


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = return_model()
model.fit(x_train, y_train, epochs=1000, batch_size=256, validation_data=(x_test,y_test))

train_pred = model.predict(x_train)
test_pred = model.predict(x_test)
print("Train Result Is ======================")
print("max_MSE ==")
print(mean_squared_error(y_train[:,0], train_pred[:,0]))
print("max_MAE ==")
print(mean_absolute_error(y_train[:,0], train_pred[:,0]))
print("min_MSE ==")
print(mean_squared_error(y_train[:,1], train_pred[:,1]))
print("min_MAE ==")
print(mean_absolute_error(y_train[:,1], train_pred[:,1]))
print("mean_MSE ==")
print(mean_squared_error(y_train[:,2], train_pred[:,2]))
print("mean_MAE ==")
print(mean_absolute_error(y_train[:,2], train_pred[:,2]))
print("Test Result Is ======================")
print("max_MSE ==")
print(mean_squared_error(y_test[:,0], test_pred[:,0]))
print("max_MAE ==")
print(mean_absolute_error(y_test[:,0], test_pred[:,0]))
print("min_MSE ==")
print(mean_squared_error(y_test[:,1], test_pred[:,1]))
print("min_MAE ==")
print(mean_absolute_error(y_test[:,1], test_pred[:,1]))
print("mean_MSE ==")
print(mean_squared_error(y_test[:,2], test_pred[:,2]))
print("mean_MAE ==")
print(mean_absolute_error(y_test[:,2], test_pred[:,2]))


def return_model2():
    input_tens = tf.keras.Input(shape=(3))
    x = tf.keras.layers.Dense(16)(input_tens)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(16)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(16)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=input_tens, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError(), metrics=["mae"])
    print(model.summary())
    return model



x_parts_2=[]
for idx in range(len(x_18parts)):
    if idx==0:
        x_parts_2.append(np.zeros(37))
    else:
        diff_raw=x_18parts[idx]-x_18parts[idx-1]
        x_parts_2.append(diff_raw)
x_parts_2 = np.array(x_parts_2, dtype=np.float32)

x_18parts = np.concatenate((x_18parts, x_parts_2), axis=1)

dq = deque(maxlen=6)
x=[]
for idx in range(len(x_18parts)):
    dq.append(x_18parts[idx])
    if len(dq)==6:
        temp_ = np.array(dq, dtype=np.float32)
        temp_ = (temp_-temp_mean)/temp_std
        x.append(temp_)
x = np.array(x)
x = x.reshape(-1,6,37,1)
new_input = model.predict(x)

new_input_train = new_input[:-144]
y_18parts_train = y_18parts[:-144]

new_input_test = new_input[-144:]
y_18parts_test = y_18parts[-144:]

model2 = return_model2()
model2.fit(new_input_train, y_18parts_train, epochs=2000, batch_size=256, validation_data=(new_input_test,y_18parts_test))

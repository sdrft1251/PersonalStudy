import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import deque
import datetime
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

train_df = pd.read_csv("./data/train.csv")
train_df.drop(columns=["X14","X16","X19"], inplace=True)
print(train_df.shape)

set1 = np.array(train_df[["X00","X07","X28","X31","X32"]], dtype=np.float32)
set2 = np.array(train_df[["X01","X06","X22","X27","X29"]], dtype=np.float32)
set3 = np.array(train_df[["X02","X03","X18","X24","X26"]], dtype=np.float32)
set4 = np.array(train_df[["X04","X10","X21","X36","X39"]], dtype=np.float32)
set5 = np.array(train_df[["X05","X08","X09","X23","X33"]], dtype=np.float32)
set6 = np.array(train_df[["X11","X34"]], dtype=np.float32)
set7 = np.array(train_df[["X12","X20","X30","X37","X38"]], dtype=np.float32)
set8 = np.array(train_df[["X13","X15","X17","X25","X35"]], dtype=np.float32)


x_parts = np.concatenate((set1,set2,set3,set4,set5,set6,set7,set8), axis=1)

x_parts_1 = x_parts[:-432]
x_parts_2 = x_parts[-432:]

y_parts = np.array(train_df[train_df.columns[38:-1]].iloc[:-432], dtype=np.float32)
y_parts_1 = y_parts
y_parts_2 = np.array(train_df[train_df.columns[-1:]].iloc[-432:], dtype=np.float32).reshape(-1)
print(x_parts_1.shape, x_parts_2.shape, y_parts_1.shape, y_parts_2.shape)

temp = x_parts_1[:-144*5]
temp_mean = temp.mean(axis=0)
temp_std = temp.std(axis=0)

x_train = x_parts_1[:-144*3]
x_test = x_parts_1[-144*3:]
y_train = y_parts_1[:-144*3]
y_test = y_parts_1[-144*3:]

x_train = (x_train-temp_mean)/temp_std
x_test = (x_test-temp_mean)/temp_std
x_parts_2_cop = x_parts_2
x_parts_2 = (x_parts_2-temp_mean)/temp_std
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

def cnn_module(input_tens, filter_list, kernel_size_num, name_idx=0):
    temp_idx = 0
    x = tf.keras.layers.Reshape((input_tens.shape[1],1))(input_tens)
    x = tf.keras.layers.Conv1D(filter_list[0], kernel_size=kernel_size_num, padding='same', name="conv_"+str(name_idx)+"_"+str(temp_idx))(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    temp_idx+=1
    for num in filter_list[1:]:
        x1 = x
        x = tf.keras.layers.Conv1D(num, kernel_size=kernel_size_num, padding='same', name="conv_"+str(name_idx)+"_"+str(temp_idx))(x)
        #x = tf.keras.layers.BatchNormalization()(x)
        temp_idx+=1
        x = tf.keras.layers.Conv1D(num, kernel_size=1, name="conv_"+str(name_idx)+"_"+str(temp_idx))(x)
        temp_idx+=1
        x = tf.keras.layers.Conv1D(num, kernel_size=1, name="conv_"+str(name_idx)+"_"+str(temp_idx))(x)
        x = x1 + x
        x = tf.keras.layers.ReLU()(x)
        temp_idx+=1
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    return x

def cnn_module2(input_tens, filter_list, kernel_size_num, name_idx=0):
    temp_idx = 0
    x = tf.keras.layers.Reshape((input_tens.shape[1],1))(input_tens)
    x = tf.keras.layers.Conv1D(filter_list[0], kernel_size=kernel_size_num, padding='same', name="conv_"+str(name_idx)+"_"+str(temp_idx))(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    temp_idx+=1
    x = tf.keras.layers.Conv1D(filter_list[1], kernel_size=kernel_size_num, padding='same', name="conv_"+str(name_idx)+"_"+str(temp_idx))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    return x

def return_model():
    input_tens = tf.keras.Input(shape=(37))
    tensor1,tensor2,tensor3,tensor4,tensor5,tensor6,tensor7,tensor8 = tf.split(input_tens, [5,5,5,5,5,2,5,5], axis=1)

    x1 = cnn_module(input_tens=tensor1, filter_list=[18,18,18,18], kernel_size_num=5, name_idx=0)
    x2 = cnn_module2(input_tens=tensor2, filter_list=[18,18], kernel_size_num=5, name_idx=1)
    x3 = cnn_module2(input_tens=tensor3, filter_list=[18,18], kernel_size_num=5, name_idx=2)
    x4 = cnn_module2(input_tens=tensor4, filter_list=[18,18], kernel_size_num=5, name_idx=3)
    x5 = cnn_module2(input_tens=tensor5, filter_list=[18,18], kernel_size_num=5, name_idx=4)
    x6 = cnn_module2(input_tens=tensor6, filter_list=[18,18], kernel_size_num=2, name_idx=5)
    x7 = cnn_module2(input_tens=tensor7, filter_list=[18,18], kernel_size_num=5, name_idx=6)
    x8 = cnn_module2(input_tens=tensor8, filter_list=[18,18], kernel_size_num=5, name_idx=7)
    x = tf.keras.layers.Add()([x1,x2,x3,x4,x5,x6,x7,x8])
    #x = tf.keras.layers.Dense(18)(x)
    model = tf.keras.Model(inputs=input_tens, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError(), metrics=["mae"])
    print(model.summary())
    return model


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = return_model()

checkpoint_path = "./temp_cp/test.ckpt"
cp_callback_best = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,monitor="val_loss",save_weights_only=True,verbose=0,save_best_only=True)
log_dir = "./test_logs/{}_".format(datetime.datetime.now())
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(x_train, y_train, epochs=1000, batch_size=256, validation_data=(x_test,y_test),callbacks=[cp_callback_best, tensorboard_callback])


print("Normal Model ~~~~~~~~~~~~~~~~~~~~~~~~~~")
train_pred = model.predict(x_train)
test_pred = model.predict(x_test)
print("Train Result Is ======================")
print("Total MSE = {} & MAE = {}".format(mean_squared_error(y_train, train_pred), mean_absolute_error(y_train, train_pred)))
print("Test Result Is ======================")
print("Total MSE = {} & MAE = {}".format(mean_squared_error(y_test, test_pred), mean_absolute_error(y_test, test_pred)))

print("Best Model ~~~~~~~~~~~~~~~~~~~~~~~~~~")
model.load_weights(checkpoint_path)
train_pred = model.predict(x_train)
test_pred = model.predict(x_test)
print("Train Result Is ======================")
print("Total MSE = {} & MAE = {}".format(mean_squared_error(y_train, train_pred), mean_absolute_error(y_train, train_pred)))
print("Test Result Is ======================")
print("Total MSE = {} & MAE = {}".format(mean_squared_error(y_test, test_pred), mean_absolute_error(y_test, test_pred)))


def return_model2():
    input_tens = tf.keras.Input(shape=(37+18))
    x = tf.keras.layers.Dense(32,activation="relu")(input_tens)
    x = tf.keras.layers.Dense(32,activation="relu")(x)
    x = tf.keras.layers.Dense(32,activation="relu")(x)
    x = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=input_tens, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError(), metrics=["mae"])
    print(model.summary())
    return model

new_input = model.predict(x_parts_2)

new_input = np.concatenate((x_parts_2_cop, new_input), axis=1)

new_input_train = new_input[:-72]
y_18parts_train = y_parts_2[:-72]

new_input_test = new_input[-72:]
y_18parts_test = y_parts_2[-72:]

model2 = return_model2()

checkpoint_path2 = "./temp_cp/test2.ckpt"
cp_callback_best2 = tf.keras.callbacks.ModelCheckpoint(checkpoint_path2,monitor="val_loss",save_weights_only=True,verbose=0,save_best_only=True)
model2.fit(new_input_train, y_18parts_train, epochs=2000, batch_size=256, validation_data=(new_input_test,y_18parts_test), callbacks=[cp_callback_best2])


print("Normal Model ~~~~~~~~~~~~~~~~~~~~~~~~~~")
train_pred = model2.predict(new_input_train)
test_pred = model2.predict(new_input_test)
print("Train Result Is ======================")
print("Total MSE = {} & MAE = {}".format(mean_squared_error(y_18parts_train, train_pred), mean_absolute_error(y_18parts_train, train_pred)))
print("Test Result Is ======================")
print("Total MSE = {} & MAE = {}".format(mean_squared_error(y_18parts_test, test_pred), mean_absolute_error(y_18parts_test, test_pred)))

print("Best Model ~~~~~~~~~~~~~~~~~~~~~~~~~~")
model2.load_weights(checkpoint_path2)
train_pred = model2.predict(new_input_train)
test_pred = model2.predict(new_input_test)
print("Train Result Is ======================")
print("Total MSE = {} & MAE = {}".format(mean_squared_error(y_18parts_train, train_pred), mean_absolute_error(y_18parts_train, train_pred)))
print("Test Result Is ======================")
print("Total MSE = {} & MAE = {}".format(mean_squared_error(y_18parts_test, test_pred), mean_absolute_error(y_18parts_test, test_pred)))
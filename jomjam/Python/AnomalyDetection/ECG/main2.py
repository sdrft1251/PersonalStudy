import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from src import vrae, train, utils
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

batch_size = 48
#time_size = 256*5
train_set = np.load('/works/Data/wellysis/preprocessed/5s_72_wellysis_stdscale_train.npy')
print("Total Data Set shape is : {}".format(train_set.shape))
time_size = train_set.shape[1]
print(train_set.min(), train_set.max(), train_set.mean(), train_set.std())

from src import IFEMBD

model = IFEMBD.IFEMBD(time_len=640, d_model=128, enc_layer_num=4, num_heads=4, dec_layer_num=4, dropout=0.1)
model.summary()

model.load_weights("/works/ModelBackup/wellysis_ai2_save/20210913_09_33_06_5sWellysis_IFEMBD_128dmodel4head_4enc_4dec_lr1e4/save")

train_loss_results = IFEMBD.train(model=model, train_set=train_set, epochs=40000, batch_size=batch_size,\
beta_cycle=0, beta_rate=0, learning_rate=1e-5,\
summary_dir= "/works/ModelBackup/wellysis_ai2_logs",\
#"/works/GitLab/jomjam/Python/AnomalyDetection/ECG/logs"
add_name="_5sWellysis_IFEMBD_128dmodel4head_4enc_4dec_lr1e5",\
cp_dir="/works/ModelBackup/wellysis_ai2_save",\
sample_data_set=train_set[:batch_size])
#"/works/GitLab/jomjam/Python/AnomalyDetection/ECG/save"
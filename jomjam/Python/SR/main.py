from sisr import data
from sisr import model
from sisr import train
import os
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
srgan_trainner = train.srgantrainer(gen_model="edsr", num_res_blocks_gen=128, num_filters_gen=384,disc_model="resnet32", content_loss_layer_num=-4,batch_size=96,\
    img_size=24, scale=4,log_save_dir="./logs/save_log", checkpoint_dir="./cp/save_cp", pre_train=False,\
    model_reuse_path=("./cp/save_cp_1583214006.2489002_w_gen","./cp/save_cp_1583214006.2489002_w_disc"), cost_rate=0.00001)

srgan_trainner.train2(data_dir="/home/temp_test/291/HR", epochs=200000)
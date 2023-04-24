import time
import tensorflow as tf
from sisr import utils
from sisr import data
import numpy as np



def EDSR(img_size=32, num_res_blocks=32,num_filters=256, res_block_scaling=0.1, scale=2):
    x_in = tf.keras.Input(shape=(img_size, img_size, 3))
    x = tf.cast(x_in, dtype=tf.float32)
    x = tf.keras.layers.Lambda(utils.normalize_02)(x)
    x = b = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = utils.res_block(b, num_filters, res_block_scaling)
    b = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(b)
    x = tf.keras.layers.Add()([x, b])
    x = utils.upsample(x, scale, num_filters)
    x = tf.keras.layers.Conv2D(3, 3, padding='same')(x)
    output = tf.keras.layers.Lambda(utils.denormalize_02)(x)
    model = tf.keras.Model(inputs=x_in, outputs=output)
    print(model.summary())
    return model


def resnet50(in_shape=None):
    x_in = tf.keras.Input(shape=in_shape)
    x = tf.cast(x_in, dtype=tf.float32)
    x = tf.keras.layers.Lambda(utils.normalize_02)(x)
    tensor = tf.keras.layers.Conv2D(64, kernel_size=(7,7), strides=(2, 2), padding='VALID')(x)
    tensor = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(tensor)
    tensor = utils.res_block_forresnet(tensor, ch_size=(64, 256), same_ch_with_before=False)
    tensor = utils.res_block_forresnet(tensor, ch_size=(64, 256), same_ch_with_before=True)
    tensor = utils.res_block_forresnet(tensor, ch_size=(64, 256), same_ch_with_before=True)
    tensor = utils.res_block_forresnet(tensor, ch_size=(128, 512), same_ch_with_before=False)
    tensor = utils.res_block_forresnet(tensor, ch_size=(128, 512), same_ch_with_before=True)
    tensor = utils.res_block_forresnet(tensor, ch_size=(128, 512), same_ch_with_before=True)
    tensor = utils.res_block_forresnet(tensor, ch_size=(128, 512), same_ch_with_before=True)
    tensor = utils.res_block_forresnet(tensor, ch_size=(256, 1024), same_ch_with_before=False)
    tensor = utils.res_block_forresnet(tensor, ch_size=(256, 1024), same_ch_with_before=True)
    tensor = utils.res_block_forresnet(tensor, ch_size=(256, 1024), same_ch_with_before=True)
    tensor = utils.res_block_forresnet(tensor, ch_size=(256, 1024), same_ch_with_before=True)
    tensor = utils.res_block_forresnet(tensor, ch_size=(256, 1024), same_ch_with_before=True)
    tensor = utils.res_block_forresnet(tensor, ch_size=(256, 1024), same_ch_with_before=True)
    tensor = utils.res_block_forresnet(tensor, ch_size=(512, 2048), same_ch_with_before=False)
    tensor = utils.res_block_forresnet(tensor, ch_size=(512, 2048), same_ch_with_before=True)
    tensor = utils.res_block_forresnet(tensor, ch_size=(512, 2048), same_ch_with_before=True)
    tensor = tf.keras.layers.AveragePooling2D(pool_size=(tensor.shape[1],tensor.shape[2]))(tensor)
    tensor = tf.keras.layers.Flatten()(tensor)
    tensor = tf.keras.layers.Dense(1, activation="sigmoid")(tensor)
    model = tf.keras.Model(inputs=x_in, outputs=tensor)
    print(model.summary())
    return model

def resnet32(in_shape=None):
    x_in = tf.keras.Input(shape=in_shape)
    x = tf.cast(x_in, dtype=tf.float32)
    x = tf.keras.layers.Lambda(utils.normalize_02)(x)
    tensor = tf.keras.layers.Conv2D(64, kernel_size=(7,7), strides=(2, 2), padding='VALID')(x)
    tensor = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(tensor)
    tensor = utils.res_block_forresnet(tensor, ch_size=(64, 256), same_ch_with_before=False)
    tensor = utils.res_block_forresnet(tensor, ch_size=(64, 256), same_ch_with_before=True)
    tensor = utils.res_block_forresnet(tensor, ch_size=(64, 256), same_ch_with_before=True)
    tensor = utils.res_block_forresnet(tensor, ch_size=(128, 512), same_ch_with_before=False)
    tensor = utils.res_block_forresnet(tensor, ch_size=(128, 512), same_ch_with_before=True)
    tensor = utils.res_block_forresnet(tensor, ch_size=(128, 512), same_ch_with_before=True)
    tensor = utils.res_block_forresnet(tensor, ch_size=(128, 512), same_ch_with_before=True)
    tensor = utils.res_block_forresnet(tensor, ch_size=(256, 1024), same_ch_with_before=False)
    tensor = utils.res_block_forresnet(tensor, ch_size=(256, 1024), same_ch_with_before=True)
    tensor = utils.res_block_forresnet(tensor, ch_size=(256, 1024), same_ch_with_before=True)
    tensor = tf.keras.layers.AveragePooling2D(pool_size=(tensor.shape[1],tensor.shape[2]))(tensor)
    tensor = tf.keras.layers.Flatten()(tensor)
    tensor = tf.keras.layers.Dense(1, activation="sigmoid")(tensor)
    model = tf.keras.Model(inputs=x_in, outputs=tensor)
    print(model.summary())
    return model

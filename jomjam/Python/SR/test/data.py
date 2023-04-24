import random
import numpy as np
import os
import tensorflow as tf



def load_dataset(data_dir):
    images=[]
    img_files = os.listdir(data_dir)
    png_files = [file for file in img_files if file.endswith(".png")]
    for file_ in png_files:
        img = tf.io.read_file(data_dir+"/"+file_)
        img = tf.image.decode_jpeg(img, channels=3)
        images.append(img)
    return images


def return_croped_images(image_list, img_size=32, scale=2):
    target_croped_image_list=[]
    x_croped_image_list=[]
    for image in image_list:
        start_w = np.random.randint(image.shape[0] - img_size*scale, size=1)[0]
        start_h = np.random.randint(image.shape[1] - img_size*scale, size=1)[0]
        y_croped_image = tf.slice(image, begin=[start_w,start_h,0],size=[img_size*scale,img_size*scale,3])
        x_croped_image = tf.image.resize(y_croped_image, (img_size, img_size), method=tf.image.ResizeMethod.BICUBIC)

        x_croped_image_list.append(x_croped_image)
        target_croped_image_list.append(y_croped_image)
    return x_croped_image_list, target_croped_image_list


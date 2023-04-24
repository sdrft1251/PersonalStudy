import random
import numpy as np
import os
import tensorflow as tf



def load_dataset(data_dir):
    images=[]
    img_files = os.listdir(data_dir)
    png_files = [file for file in img_files if file.endswith((".png",".bmp",".jpg"))]
    for file_ in png_files:
        if file_[-3:]=="bmp":
            img = tf.io.read_file(data_dir+"/"+file_)
            img = tf.io.decode_bmp(img, channels=3)
            images.append(img)
        elif file_[-3:]=="png":
            img = tf.io.read_file(data_dir+"/"+file_)
            img = tf.io.decode_png(img, channels=3)
            images.append(img)
        else:
            img = tf.io.read_file(data_dir+"/"+file_)
            img = tf.io.decode_jpeg(img, channels=3)
            images.append(img)
    return images


def return_croped_images(image_list, img_size=32, scale=2):
    target_croped_image_list=[]
    x_croped_image_list=[]
    for image in image_list:
        start_w = np.random.randint(low=0,high=image.shape[0] - img_size*scale, size=1)[0]
        start_h = np.random.randint(low=0,high=image.shape[1] - img_size*scale, size=1)[0]
        y_croped_image = tf.slice(image, begin=[start_w,start_h,0],size=[img_size*scale,img_size*scale,3])
        x_croped_image = tf.image.resize(y_croped_image, (img_size, img_size), method=tf.image.ResizeMethod.BICUBIC)

        x_croped_image_list.append(x_croped_image)
        target_croped_image_list.append(y_croped_image)
    return x_croped_image_list, target_croped_image_list


def return_test_images(image, scale=2):
    target_img = tf.reshape(image,(1, image.shape[0], image.shape[1], image.shape[2]))
    input_img = tf.reshape(\
        tf.image.resize(\
            image, (int(image.shape[0]/scale), int(image.shape[1]/scale)), method=tf.image.ResizeMethod.BICUBIC),\
                (1, int(image.shape[0]/scale), int(image.shape[1]/scale), image.shape[2]))
    for_comparing = tf.reshape(tf.image.resize(input_img, (image.shape[0], image.shape[1]), method=tf.image.ResizeMethod.BICUBIC),(1, image.shape[0], image.shape[1], image.shape[2]))

    return target_img, input_img, for_comparing
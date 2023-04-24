import numpy as np
import tensorflow as tf


DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255


def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch


def evaluate(model, dataset):
    psnr_values = []
    for lr, hr in dataset:
        sr = resolve(model, lr)
        psnr_value = psnr(hr, sr)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)

# ---------------------------------------
#  Normalization
# ---------------------------------------


def normalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return (x - rgb_mean) / 127.5


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return x * 127.5 + rgb_mean


def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0

def denormalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x * 255.0

def normalize_02(x):
    """Normalizes RGB images to [0, 2]."""
    return x / 127.5

def denormalize_02(x):
    """Normalizes RGB images to [0, 2]."""
    return x * 127.5


def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5


# ---------------------------------------
#  Metrics
# ---------------------------------------


def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)


# ---------------------------------------
#  See https://arxiv.org/abs/1609.05158
# ---------------------------------------


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


# ---------------------------------------
#  Block
# ---------------------------------------

def res_block(x_in, filters, scaling):
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = tf.keras.layers.Lambda(lambda t: t * scaling)(x)
    x = tf.keras.layers.Add()([x_in, x])
    return x


def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        x = tf.keras.layers.Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return tf.keras.layers.Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')

    return x

def res_block_forresnet(x, ch_size=(64, 256), same_ch_with_before=True):
    if not same_ch_with_before:
        out = x
        x = tf.keras.layers.Conv2D(ch_size[1], kernel_size=(1,1), padding="same")(x)
    else:
        out = x
    tensor = tf.keras.layers.Conv2D(ch_size[0], kernel_size=(1,1), padding="same")(out)
    tensor = tf.keras.layers.Conv2D(ch_size[0], kernel_size=(3,3), padding="same")(tensor)
    tensor = tf.keras.layers.Conv2D(ch_size[1], kernel_size=(1,1), padding="same")(tensor)
    tensor = tensor + x
    return tensor
    
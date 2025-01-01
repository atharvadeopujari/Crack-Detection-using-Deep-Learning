import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
from keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")
print(mixed_precision.global_policy())

def one_hot_encode_mask(mask, num_classes=2):

    mask = tf.cast(mask, tf.int32)
    return tf.one_hot(mask, depth=num_classes, axis=-1)

# Data augmentation
def data_aug(img, mask):

    f = tf.image.resize(img, size = (2048//4, 2560//4))
    g = tf.image.resize(mask, size = (2048//4, 2560//4))

    # Random Flipping
    k = tf.random.uniform(shape=[3], minval=0, maxval=1, dtype=tf.float32)
    if k[0] < 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)

    if k[1] < 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)

    return f, g


# Validation data
def val_data(img, mask):

    f = tf.image.resize(img, size = (2048//4, 2560//4))
    g = tf.image.resize(mask, size = (2048//4, 2560//4))

    return f, g

# Reading image and mask files
def img_read_fun(filename):
    
    img = tf.io.decode_png(tf.io.read_file(filename))
    
    # Normalize pixel values to [0, 1]
    img = tf.image.convert_image_dtype(img, tf.float32)

    return img

def mask_read_fun(filename):

    mask = tf.io.decode_png(tf.io.read_file(filename))

    # Normalize pixel values to [0, 1]
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    
    # Ensure binary values (0 or 1)
    mask = tf.math.round(mask)

    # One-hot encode the mask
    mask = one_hot_encode_mask(mask, num_classes=2)
    
    return mask

# Dataset Pipeline
def create_datasets(first_img_dir, first_img_mask_dir, second_img_dir, second_img_mask_dir, train_batch_size):
    
    # img
    first_ds_img = tf.data.Dataset.list_files(first_img_dir + "*.png", shuffle = False)
    first_ds_img = first_ds_img.map(img_read_fun, num_parallel_calls = tf.data.AUTOTUNE)

    second_ds_img = tf.data.Dataset.list_files(second_img_dir + "*.png", shuffle = False)
    second_ds_img = second_ds_img.map(img_read_fun, num_parallel_calls = tf.data.AUTOTUNE)
    
    # mask
    first_ds_mask = tf.data.Dataset.list_files(first_img_mask_dir + "*.png", shuffle = False)
    first_ds_mask = first_ds_mask.map(mask_read_fun, num_parallel_calls = tf.data.AUTOTUNE)

    second_ds_mask = tf.data.Dataset.list_files(second_img_mask_dir + "*.png", shuffle = False)
    second_ds_mask = second_ds_mask.map(mask_read_fun, num_parallel_calls = tf.data.AUTOTUNE)

    # pairing the images and masks
    first_ds = tf.data.Dataset.zip((first_ds_img, first_ds_mask))
    second_ds = tf.data.Dataset.zip((second_ds_img, second_ds_mask))

    #combine and shuffle
    ds = first_ds.concatenate(second_ds)
    ds = ds.shuffle(buffer_size = 2500)

    # 80% train, 20% validation
    train_ds = ds.take(1612)
    val_ds = ds.skip(1612).take(403)

    return train_ds, val_ds
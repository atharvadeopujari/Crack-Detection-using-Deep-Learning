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

    # Tip
    idx = tf.where(g)

    x = tf.range(0, 2560//4, dtype = tf.float32)
    x = tf.repeat(x, 2048//4, axis = 0)
    x = tf.reshape(x, shape = (2560//4, 2048//4))
    x = tf.transpose(x)

    y = tf.range(0, 2048//4, dtype = tf.float32)
    y = tf.repeat(y, 2560//4, axis = 0)
    y = tf.reshape(y, shape = (2048//4, 2560//4))
    
    
    mx = idx[0, 0]
    my = idx[0, 1]

    mx = tf.cast(mx, dtype = tf.float32)
    my = tf.cast(my, dtype = tf.float32)

    var = 1e4

    t = tf.math.exp(-(tf.math.pow((x - my), 2) + tf.math.pow((y - mx), 2))/(var))   
    t = tf.reshape(t, shape = (2048//4, 2560//4, 1))

    # Random Flipping
    k = tf.random.uniform(shape=[3], minval=0, maxval=1, dtype=tf.float32)
    if k[0] < 0.5:
        f = tf.image.flip_left_right(f)
        g = tf.image.flip_left_right(g)
        t = tf.image.flip_left_right(t)

    if k[1] < 0.5:
        f = tf.image.flip_up_down(f)
        g = tf.image.flip_up_down(g)
        t = tf.image.flip_up_down(t)

    return f, g, t


# Validation data
def val_data(img, mask):

    f = tf.image.resize(img, size = (2048//4, 2560//4))
    g = tf.image.resize(mask, size = (2048//4, 2560//4))

    idx = tf.where(g)

    x = tf.range(0, 2560//4, dtype = tf.float32)
    x = tf.repeat(x, 2048//4, axis = 0)
    x = tf.reshape(x, shape = (2560//4, 2048//4))
    x = tf.transpose(x)

    y = tf.range(0, 2048//4, dtype = tf.float32)
    y = tf.repeat(y, 2560//4, axis = 0)
    y = tf.reshape(y, shape = (2048//4, 2560//4))
    
    
    mx = idx[0, 0]
    my = idx[0, 1]

    mx = tf.cast(mx, dtype = tf.float32)
    my = tf.cast(my, dtype = tf.float32)

    var = 1e4

    t = tf.math.exp(-(tf.math.pow((x - my), 2) + tf.math.pow((y - mx), 2))/(var))   
    t = tf.reshape(t, shape = (2048//4, 2560//4, 1))

    return f, g, t

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
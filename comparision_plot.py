import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
from keras import mixed_precision
import matplotlib.pyplot as plt 
from PIL import Image
from dataset import create_datasets, data_aug, val_data


mixed_precision.set_global_policy("mixed_float16")
print(mixed_precision.global_policy())

# Dataset Directories
first_img_dir = "C:\\Users\\user\\Desktop\\Atharva Deopujari\\CIT_data_1500\\DCW_1500\\"
first_img_mask_dir = "C:\\Users\\user\\Desktop\\Atharva Deopujari\\CIT_data_1500\\mask_1500\\"
second_img_dir = "C:\\Users\\user\\Desktop\\Atharva Deopujari\\CIT RB\\png\\img\\"
second_img_mask_dir = "C:\\Users\\user\\Desktop\\Atharva Deopujari\\CIT RB\\png\\mask\\"


train_batch_size = 16

train_ds, val_ds = create_datasets(first_img_dir, first_img_mask_dir, second_img_dir, second_img_mask_dir, train_batch_size)
print("training size = ", train_ds.cardinality())
print("validation size = ", val_ds.cardinality())

train_ds = train_ds.cache()
train_ds = train_ds.repeat(4)
print("Training size = " + str(train_ds.cardinality()))
train_ds = train_ds.map(data_aug, num_parallel_calls = tf.data.AUTOTUNE)
train_ds_batched = train_ds.batch(train_batch_size)
train_ds_batched = train_ds_batched.prefetch(buffer_size = tf.data.AUTOTUNE)

# validation input pipeline
val_ds = val_ds.cache()
val_ds = val_ds.repeat(1)
print("Validation size = " + str(val_ds.cardinality()))
val_ds = val_ds.map(val_data, num_parallel_calls = tf.data.AUTOTUNE)
val_ds_batched = val_ds.batch(train_batch_size)
val_ds_batched = val_ds_batched.prefetch(buffer_size = tf.data.AUTOTUNE)
net=keras.models.load_model('trained_model.keras', compile=False)
net.summary(150)


for step, (x, y) in enumerate(train_ds_batched.take(10).as_numpy_iterator()):
    z = net.predict(x)
    img = Image.fromarray(x[0, :, :, 0]*255)
    img2 = Image.fromarray(y[0, :, :, 0]*255)
    img3 = Image.fromarray(z[0, :, :, 0]*255)

    print(x)
    print(y)
    print(z)
  
    plt.figure("Image")
    plt.imshow(img)

    plt.figure("Mask")
    plt.imshow(img2)

    plt.figure("Prediction")
    plt.imshow(img3)

    plt.show()
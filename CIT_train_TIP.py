import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
from keras import mixed_precision
import matplotlib.pyplot as plt 
import math
from random import randint
import time
from PIL import Image
from dataset import create_datasets, data_aug, val_data


mixed_precision.set_global_policy("mixed_float16")
print(mixed_precision.global_policy())

# Dataset Directories
first_img_dir = "C:\\Users\\user\\Desktop\\Atharva Deopujari\\CIT_data_1500\\DCW_1500\\"
first_img_mask_dir = "C:\\Users\\user\\Desktop\\Atharva Deopujari\\CIT_data_1500\\mask_1500\\"
second_img_dir = "C:\\Users\\user\\Desktop\\Atharva Deopujari\\CIT RB\\png\\img\\"
second_img_mask_dir = "C:\\Users\\user\\Desktop\\Atharva Deopujari\\CIT RB\\png\\mask\\"


# dataset_pipeline
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

# Model Load
Unet = keras.models.load_model('UNet_CIT_untrained.h5', compile = False)
Unet.summary(150)

##training
boundaries = [60*train_ds_batched.cardinality(), 90*train_ds_batched.cardinality()]
values = [1e-4, 1e-5, 1e-6]
learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
#opt = keras.optimizers.AdamW(learning_rate=learning_rate_fn, weight_decay=1e-4)
opt = keras.optimizers.Adam(learning_rate_fn)
opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
BCE_loss = keras.losses.BinaryCrossentropy()
BA_metric = keras.metrics.BinaryAccuracy()
Pr_metric = keras.metrics.Precision()

seg_met = keras.metrics.MeanIoU(num_classes = 2)

# Loss Fun

def dice_loss(y_true, y_pred, smooth=1):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def boundary_loss(y_true, y_pred, smooth=1e-6):
    y_true_grad = tf.image.sobel_edges(y_true)  # Approximate boundary using gradients
    y_pred_grad = tf.image.sobel_edges(y_pred)
    intersection = tf.reduce_sum(y_true_grad * y_pred_grad)
    total = tf.reduce_sum(y_true_grad) + tf.reduce_sum(y_pred_grad)
    return 1 - (2 * intersection + smooth) / (total + smooth)


def loss_fun(y, y_pred):
    
    bce_loss = keras.losses.BinaryCrossentropy()(y, y_pred)
    #bce_loss = tf.cast(bce_loss, dtype=tf.float32)
    
    dice = dice_loss(y, y_pred)

    total_loss = bce_loss + dice
    
    total_loss = tf.reduce_mean(total_loss)

    return total_loss
#bce_loss = keras.losses.binary_crossentropy(y, y_pred)
#bce_loss = keras.losses.BinaryCrossentropy()
# loss_fun = bce_loss

# Loss Function
# def loss_fun(y, y_pred):
    
#     # Compute binary cross-entropy loss
#     bce_loss = tf.keras.losses.binary_crossentropy(y, y_pred)
    
#     # Reduce the loss across all pixels
#     total_loss = tf.reduce_mean(bce_loss)
    
#     return total_loss

# Metrics
def pixel_accuracy(y_true, y_pred):

    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)  # Binarize predictions

    # Accuracy for background (class 0)
    background_true = tf.cast(tf.equal(y_true, 0), tf.float32)
    background_pred = tf.cast(tf.equal(y_pred_binary, 0), tf.float32)
    background_correct = tf.reduce_sum(background_true * background_pred)
    background_total = tf.reduce_sum(background_true)
    background_accuracy = (background_correct + 1e-7) / (background_total + 1e-7)

    # Accuracy for foreground (class 1)
    foreground_true = tf.cast(tf.equal(y_true, 1), tf.float32)
    foreground_pred = tf.cast(tf.equal(y_pred_binary, 1), tf.float32)
    foreground_correct = tf.reduce_sum(foreground_true * foreground_pred)
    foreground_total = tf.reduce_sum(foreground_true)
    foreground_accuracy = (foreground_correct + 1e-7) / (foreground_total + 1e-7)

    # Average accuracy
    total_accuracy = (background_accuracy + foreground_accuracy) / 2
    return total_accuracy

# Train Step
@tf.function(jit_compile=True)
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = Unet(x, training=True)
        loss = loss_fun(y, y_pred)
        scaled_loss = opt.get_scaled_loss(loss)
    
    
    scaled_grads = tape.gradient(scaled_loss, Unet.trainable_variables)
    grads = opt.get_unscaled_gradients(scaled_grads)  
    opt.apply_gradients(zip(grads, Unet.trainable_variables))

    loss_value = tf.reduce_mean(loss)

    seg_met.reset_state()
    seg_met.update_state(y, y_pred)
    met_value = seg_met.result()

    accuracy = pixel_accuracy(y, y_pred)

    return loss_value, met_value, accuracy


# validation step
@tf.function(jit_compile=True)
def val_step(x, y):
    y_pred = Unet(x, training=False)
    loss = loss_fun(y, y_pred)

    seg_met.reset_state()
    seg_met.update_state(y, y_pred)
    met_value = seg_met.result()

    accuracy = pixel_accuracy(y, y_pred)

    return loss, met_value, accuracy


def val():
    val_loss = np.zeros(shape = (val_ds_batched.cardinality(), 1))
    val_BA = np.zeros(shape = (val_ds_batched.cardinality(), 1))
    val_accuracy = np.zeros(shape = (val_ds_batched.cardinality(), 1))

    for step, (x, y) in enumerate(val_ds_batched):        
        val_loss[step], val_BA[step], val_accuracy[step]= val_step(x, y)

    return np.mean(val_loss), np.mean(val_BA), np.mean(val_accuracy)

# Training Loop

max_epochs = 120

epoch_train_loss = np.zeros(shape = (max_epochs, 1))
epoch_train_BA = np.zeros(shape = (max_epochs, 1))
epoch_train_accuracy = np.zeros(shape = (max_epochs, 1))
epoch_val_loss = np.zeros(shape = (max_epochs, 1))
epoch_val_BA = np.zeros(shape = (max_epochs, 1))
epoch_val_accuracy = np.zeros(shape = (max_epochs, 1))

step_train_loss = np.zeros(shape = (train_ds_batched.cardinality(), 1))
step_train_BA = np.zeros(shape = (train_ds_batched.cardinality(), 1))
step_train_accuracy = np.zeros(shape = (train_ds_batched.cardinality(), 1))

for epoch in range(max_epochs):

    start_time = time.time()

    print("Epoch " + str(epoch + 1 ) + "/" + str(max_epochs))
    print("Learning rate = ", opt.learning_rate(opt.iterations).numpy())

    for step, (x, y) in enumerate(train_ds_batched):

        step_train_loss[step], step_train_BA[step], step_train_accuracy[step] = train_step(x, y)

    epoch_train_loss[epoch] = np.mean(step_train_loss)
    epoch_train_BA[epoch] = np.mean(step_train_BA)
    epoch_train_accuracy[epoch] = np.mean(step_train_accuracy)
    print("loss = " + str(epoch_train_loss[epoch]))    
    print("mIOU = " + str(epoch_train_BA[epoch]))
    print("accuracy = " + str(epoch_train_accuracy[epoch]))

    epoch_val_loss[epoch], epoch_val_BA[epoch], epoch_val_accuracy[epoch] = val()

    
    print("\nval loss = " + str(epoch_val_loss[epoch]))    
    print("val mIOU = " + str(epoch_val_BA[epoch]))
    print("val accuracy = " + str(epoch_val_accuracy[epoch]))

    end_time = time.time()
    print("time = " + str(end_time - start_time) + "\n")


Unet.save(filepath = "trained_model_dice.h5")

np.save("train_loss.npy", epoch_train_loss)
np.save("train_accu.npy", epoch_train_BA)
np.save("train_pixel_accuracy.npy", epoch_train_accuracy)

np.save("val_loss.npy", epoch_val_loss)
np.save("val_accu.npy", epoch_val_BA)
np.save("val_pixel_accuracy.npy", epoch_val_accuracy)





import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
from keras import mixed_precision
from keras import regularizers
from keras import models
import matplotlib.pyplot as plt 
import h5py
import math
from random import randint
# import pydot
# import graphviz
import time
from PIL import Image



mixed_precision.set_global_policy("mixed_float16")
print(mixed_precision.global_policy())

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


# mode definition
# h = 2048
# w = 2560
# Attention Gate
def attention_gate(x, g, inter_channels):

    theta_x = layers.Conv2D(inter_channels, kernel_size=1, strides=1, padding="same")(x)
    phi_g = layers.Conv2D(inter_channels, kernel_size=1, strides=1, padding="same")(g)
    add = layers.Add()([theta_x, phi_g])
    relu = layers.Activation('relu')(add)
    psi = layers.Conv2D(1, kernel_size=1, strides=1, padding="same")(relu)
    psi = layers.Activation('sigmoid')(psi)
    return layers.Multiply()([x, psi])  # Multiply the input by the attention map

# Model Dimensions
h = None
w = None

# Input
img_in = layers.Input(shape=(h, w, 1))

# Encoder
x1 = layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(img_in)
x1 = layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(x1)
p1 = layers.MaxPooling2D(pool_size=2)(x1)

x2 = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(p1)
x2 = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x2)
p2 = layers.MaxPooling2D(pool_size=2)(x2)

x3 = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(p2)
x3 = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x3)
p3 = layers.MaxPooling2D(pool_size=2)(x3)

x4 = layers.Conv2D(256, kernel_size=3, padding="same", activation="relu")(p3)
x4 = layers.Conv2D(256, kernel_size=3, padding="same", activation="relu")(x4)
p4 = layers.MaxPooling2D(pool_size=2)(x4)

# Bottleneck
x5 = layers.Conv2D(512, kernel_size=3, padding="same", activation="relu")(p4)
x5 = layers.Conv2D(512, kernel_size=3, padding="same", activation="relu")(x5)

# Decoder with Attention Gates
a4 = attention_gate(x4, x5, inter_channels=128)
u4 = layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding="same", activation="relu")(x5)
u4 = layers.Concatenate()([u4, a4])
u4 = layers.Conv2D(256, kernel_size=3, padding="same", activation="relu")(u4)
u4 = layers.Conv2D(256, kernel_size=3, padding="same", activation="relu")(u4)

a3 = attention_gate(x3, u4, inter_channels=64)
u3 = layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same", activation="relu")(u4)
u3 = layers.Concatenate()([u3, a3])
u3 = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(u3)
u3 = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(u3)

a2 = attention_gate(x2, u3, inter_channels=32)
u2 = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same", activation="relu")(u3)
u2 = layers.Concatenate()([u2, a2])
u2 = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(u2)
u2 = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(u2)

a1 = attention_gate(x1, u2, inter_channels=16)
u1 = layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation="relu")(u2)
u1 = layers.Concatenate()([u1, a1])
u1 = layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(u1)
u1 = layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(u1)

# Output
img_out = layers.Conv2D(1, kernel_size=1, activation="sigmoid")(u1)

# Model
Attention_Unet = models.Model(inputs=img_in, outputs=img_out, name="Attention-UNet")
Attention_Unet.summary()
#Attention_Unet.save("Attention_Unet_untrained.h5")

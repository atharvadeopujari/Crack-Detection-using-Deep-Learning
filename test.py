import tensorflow as tf
from tensorflow import keras


a = tf.zeros(shape = (256, 256, 1))
b = tf.ones(shape = (256, 256, 1))

# bce_loss = keras.losses.binary_crossentropy(a, b) 
bce_loss = keras.losses.BinaryCrossentropy()

y = bce_loss(a, b)

print(y.shape)
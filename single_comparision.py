import numpy as np
import math
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model
import matplotlib.pyplot as plt
from random import randint
from dataset import img_read_fun, mask_read_fun


img_dir = "C:\\Users\\user\\Desktop\\Atharva Deopujari\\CIT_data_1500\\DCW_1500\\"
img_dir_mask = "C:\\Users\\user\\Desktop\\Atharva Deopujari\\CIT_data_1500\\mask_1500\\"

net = keras.models.load_model('trained_model.keras', compile=False)
net.summary()

i = 1200
img_name = img_dir + str(i) + '.png'
img_array = img_read_fun(img_name) 
x = np.zeros((1, img_array.shape[0], img_array.shape[1], 1))
x[0] = img_array

mask_name = img_dir_mask + str(i) + '.png'
img_mask_array = mask_read_fun(mask_name)
y = np.zeros_like(x)
y[0] = img_mask_array

z = net.predict(x)
z[0, :, :, 0] = (z[0, :, :, 0] > 0.5).astype(np.float32)  # Apply threshold

plt.figure(1)
plt.imshow(x[0, :, :, 0], cmap='gray')
plt.title("Input Image")

plt.figure(2)
plt.imshow(y[0, :, :, 0], cmap='gray')
plt.title("Ground Truth Mask")

plt.figure(3)
plt.imshow(z[0, :, :, 0], cmap='gray')
plt.title("Predicted Mask")

plt.show()


# no = 1

# h = 2048
# w = 2560

# x=np.zeros((no,h,w,1))
# y=np.zeros((no,h,w,1))
# z=np.zeros((no,h,w,1))

# net=keras.models.load_model('trained_model.keras', compile=False)

# net.summary(150)

# i=1200

# img_name = img_dir + str(i) + '.png'
# img = Image.open(img_name)
# img_array = np.array(img)

# mask_name = img_dir_mask + str(i) + '.png'
# img_mask = Image.open(mask_name)
# img_mask_array = np.array(img_mask)
# g = img_array
# y[0, :, :, 0] = img_mask_array
# x[0, :, :, 0] = g/255
# z = net.predict(x)

# img = Image.fromarray(x[0, :, :, 0]*255)
# plt.figure(1)
# plt.imshow(img)

# img = Image.fromarray(z[0, :, :, 0]*255)
# plt.figure(2)
# plt.imshow(img)

# img = Image.fromarray(y[0, :, :, 0]*255)
# plt.figure(3)
# plt.imshow(img)

# plt.show()


import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from random import sample
from dataset import img_read_fun, mask_read_fun

# Directories
# img_dir = "C:\\Users\\user\\Desktop\\Atharva Deopujari\\CIT_data_1500\\DCW_1500\\"
# img_dir_mask = "C:\\Users\\user\\Desktop\\Atharva Deopujari\\CIT_data_1500\\mask_1500\\"
img_dir = "C:\\Users\\user\\Desktop\\Atharva Deopujari\\CIT RB\\png\\img\\"
img_dir_mask = "C:\\Users\\user\\Desktop\\Atharva Deopujari\\CIT RB\\png\\mask\\"

# Load the model
net = keras.models.load_model('trained_model_v2.h5', compile=False)
net.summary()

total_images = 516
random_indices = sample(range(total_images), 100)

for i in random_indices:
    
    img_name = img_dir + str(i) + '.png'
    img_array = img_read_fun(img_name) 

    img_array = tf.image.resize(img_array, size = (512, 640))
    x = tf.reshape(img_array, shape = (1, 512, 640, 1))


    mask_name = img_dir_mask + str(i) + '.png'
    img_mask_array = mask_read_fun(mask_name)
  
    img_mask_array = tf.image.resize(img_mask_array, size = (512, 640))
    y = tf.reshape(img_mask_array, shape = (1, 512, 640, 1))
  
    z = net.predict(x)
    #z[0, :, :, 0] = (z[0, :, :, 0] > 0.5).astype(np.float32)  # Apply threshold

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(x[0, :, :, 0], cmap = "gray")
    plt.title(f"Input Image {i}")

    plt.subplot(1, 3, 2)
    plt.imshow(y[0, :, :, 0])
    plt.title(f"Ground Truth Mask {i}")

    plt.subplot(1, 3, 3)
    plt.imshow(z[0, :, :, 0])
    plt.title(f"Predicted Mask {i}")

    # img = Image.fromarray(x[:, :, 0])
    # plt.imshow(img)

    # img = Image.fromarray(y[:, :, 0])
    # plt.imshow(img)

    # img = Image.fromarray(z[:, :, 0])
    # plt.imshow(img)

    plt.show()

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from random import sample
from dataset import img_read_fun, mask_read_fun

# Directories
img_dir = "C:\\Users\\user\\Desktop\\Atharva Deopujari Scratch\\val data\\img\\"
img_dir_mask = "C:\\Users\\user\\Desktop\\Atharva Deopujari Scratch\\val data\\mask\\"
output_dir = "C:\\Users\\user\\Desktop\\Atharva Deopujari Scratch\\val data\\predicted\\"

# Load the model
net = keras.models.load_model('trained_model_v1.h5', compile=False)
net.summary()

# total images -> 403 (20% of 2015)
# old dataset -> 414 - 516
# new dataset -> 1201 - 1501
#random_indices = sample(range(414,517),103) + sample(range(1201,1502),300)

#for i in random_indices:
for i in list(range(414, 517)) + list(range(1201, 1502)):
    
    img_name = img_dir + str(i) + '.png'
    img_array = img_read_fun(img_name) 

    img_array = tf.image.resize(img_array, size = (512, 640))
    x = tf.reshape(img_array, shape = (1, 512, 640, 1))


    mask_name = img_dir_mask + str(i) + '.png'
    img_mask_array = mask_read_fun(mask_name)
  
    img_mask_array = tf.image.resize(img_mask_array, size = (512, 640))
    y = tf.reshape(img_mask_array, shape = (1, 512, 640, 1))
  
    z = net.predict(x)
    predicted_mask = z[0, :, :, 0] > 0.5

    # topmost pixel with value 1
    coords = np.argwhere(predicted_mask)
    ymin, xmin = coords[0]

    zoom_size = 50 
    ymin_start = max(ymin - zoom_size // 2, 0)
    xmin_start = max(xmin - zoom_size // 2, 0)
    ymin_end = min(ymin + zoom_size // 2, predicted_mask.shape[0])
    xmin_end = min(xmin + zoom_size // 2, predicted_mask.shape[1])

    # Crop the images and masks
    cropped_input = x[0, ymin_start:ymin_end, xmin_start:xmin_end, 0]
    cropped_ground_truth = y[0, ymin_start:ymin_end, xmin_start:xmin_end, 0]
    cropped_predicted_mask = predicted_mask[ymin_start:ymin_end, xmin_start:xmin_end]

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(x[0, :, :, 0], cmap = "gray")
    plt.title(f"Input Image {i}")

    plt.subplot(1, 4, 2)
    plt.imshow(y[0, :, :, 0])
    plt.title(f"Ground Truth Mask {i}")

    plt.subplot(1, 4, 3)
    plt.imshow(z[0, :, :, 0])
    plt.title(f"Predicted Mask {i}")

    plt.subplot(1, 4, 4)
    plt.imshow(x[0, :, :, 0], cmap="gray") 
    plt.imshow(z[0, :, :, 0], cmap="jet", alpha=0.5)  # Overlay with transparency
    plt.title(f"Overlay {i}")

    plt.show()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(x[0, :, :, 0], cmap="gray")
    ax.imshow(z[0, :, :, 0], cmap="jet", alpha=0.5)  # Overlay with transparency
    ax.set_title(f"Overlay {i}")
    # save_path = os.path.join(output_dir, f"{i}.png")
    # fig.savefig(save_path)
    # plt.close(fig)
    # plt.savefig(save_path)
    # plt.close()

    plt.show()

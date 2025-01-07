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
bce_model = keras.models.load_model('trained_model_bce.h5', compile=False)
dice_model = keras.models.load_model('trained_model_dice', compile=False)
focal_model = keras.models.load_model('trained_model_focal', compile=False)
tversky_model = keras.models.load_model('trained_model_tversky', compile=False)

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
  
    bce_pred = bce_model.predict(x)[0, :, :, 0]
    dice_pred = dice_model.predict(x)[0, :, :, 0]
    focal_pred = focal_model.predict(x)[0, :, :, 0]
    tversky_pred = tversky_model.predict(x)[0, :, :, 0]

    # Plot comparison
    plt.figure(figsize=(12, 8))

    # Input Image
    plt.subplot(2, 3, 1)
    plt.imshow(x[0, :, :, 0], cmap="gray")
    plt.title(f"Input Image {i}")

    # Ground Truth
    plt.subplot(2, 3, 2)
    plt.imshow(y[0, :, :, 0], cmap="gray")
    plt.title("Ground Truth")

    # BCE Model Prediction
    plt.subplot(2, 3, 3)
    plt.imshow(bce_pred, cmap="gray")
    plt.title("BCE Prediction")

    # Dice Model Prediction
    plt.subplot(2, 3, 4)
    plt.imshow(dice_pred, cmap="gray")
    plt.title("Dice Prediction")

    # Focal Model Prediction
    plt.subplot(2, 3, 5)
    plt.imshow(focal_pred, cmap="gray")
    plt.title("Focal Prediction")

    # Tversky Model Prediction
    plt.subplot(2, 3, 6)
    plt.imshow(tversky_pred, cmap="gray")
    plt.title("Tversky Prediction")

    plt.tight_layout()
    plt.show()

    # save_path = os.path.join(output_dir, f"{i}.png")
    # fig.savefig(save_path)
    # plt.close(fig)
    # plt.savefig(save_path)
    # plt.close()

    plt.show()

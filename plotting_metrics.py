import matplotlib.pyplot as plt
import numpy as np

train_loss_history = np.load("train_loss.npy")
val_loss_history = np.load("val_loss.npy")
train_miou_history = np.load("train_accu.npy")
val_miou_history = np.load("val_accu.npy")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(train_loss_history, label="Training Loss", color="blue")
ax1.plot(val_loss_history, label="Validation Loss", color="orange")
ax1.set_title("Loss Over Epochs")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

ax2.plot(train_miou_history, label="Training mIoU", color="green")
ax2.plot(val_miou_history, label="Validation mIoU", color="red")
ax2.set_title("Mean IoU Over Epochs")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Mean IoU")
ax2.legend()
ax2.grid(True)

plt.show()
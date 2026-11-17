import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File("data/fluvial_binary_64x64.h5", "r") as f:
    images = f["images"][:]

# pick 16 random images and show them in a 4x4 grid
idx = np.random.choice(len(images), size=16, replace=False)

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, i in zip(axes.ravel(), idx):
    ax.imshow(images[i], cmap="gray", interpolation="none", origin="upper")
    ax.axis("off")

fig.suptitle("Channel images")
plt.tight_layout()
plt.show()

plt.imshow(np.mean(images, axis=0), cmap="gray", interpolation="none", origin="upper")
plt.title("Mean")
plt.tight_layout()
plt.show()
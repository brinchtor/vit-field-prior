import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File("02_fluvial_binary_images.h5", "r") as f:
    images = f["images_channel"][:, :128, 64:192]
    images_crevasse = f["images_channel_crevasse"][:, 64:192, 64:192]

# pick 16 random images and show them in a 4x4 grid
idx = np.random.choice(len(images), size=16, replace=False)

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, i in zip(axes.ravel(), idx):
    ax.imshow(images[i], cmap="gray", interpolation="none", origin="upper")
    ax.axis("off")

fig.suptitle("Channel images")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, i in zip(axes.ravel(), idx):
    ax.imshow(images_crevasse[i], cmap="gray", interpolation="none", origin="upper")
    ax.axis("off")

fig.suptitle("Channel crevasse images")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize = (10,5))

axes[0].imshow(np.mean(images, axis=0), cmap="gray", interpolation="none", origin="upper")
axes[1].imshow(np.mean(images_crevasse, axis=0), cmap="gray", interpolation="none", origin="upper")

axes[0].axis("off")

axes[1].axis("off")

fig.suptitle("Means; crevasse right.")
plt.tight_layout()
plt.show()
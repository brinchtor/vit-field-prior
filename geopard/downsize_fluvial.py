import h5py
import numpy as np

src_path = "02_fluvial_binary_images.h5"
out_path = "data/fluvial_binary_64x64.h5"

print("Loading...")
with h5py.File(src_path, "r") as f:
    images = f["images_channel_crevasse"][:]
    seeds  = f["seeds"][:]

print("Downsampling...")
# 2x2 majority vote: reshape into 2x2 blocks and round the mean
imgs64 = np.round(
    images.reshape(len(images), 64, 4, 64, 4).mean(axis=(2, 4))
).astype(np.uint8)

print(f"Output shape: {imgs64.shape}, unique values: {np.unique(imgs64)}")

with h5py.File(out_path, "w") as f:
    f.create_dataset("images", data=imgs64,
                     chunks=(1, 64, 64),
                     compression="gzip", compression_opts=4)
    f.create_dataset("seeds", data=seeds)

print(f"Saved to {out_path}")
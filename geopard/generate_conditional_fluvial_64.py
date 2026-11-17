"""
generate_conditional_fluvial_64.py

Takes the first image in the held-out test set (last 1000 images of the full
fluvial 64x64 GEOPARD dataset), observes all pixels in columns
22 and 43 (64 pixels each),
and generates 1000 posterior samples.  Saves images together with conditional
and unconditional log-likelihoods for importance-weighted resampling.

Usage:
    python geopard/generate_conditional_fluvial_64.py

Output:  data/conditional_samples_fluvial_64.npz
    images          float32  [1000, H, W]   – posterior samples
    log_lik_cond    float32  [1000]          – log q(x | y)  (approx posterior)
    log_lik_uncond  float32  [1000]          – log p_theta(x) (ViT prior)
    observed_image  float32  [H, W]          – the test image used
    condition_cols  int32    [2]             – column indices that were observed
"""
import os as _os_shim, sys as _sys_shim
_sys_shim.path.insert(0, _os_shim.path.join(_os_shim.path.dirname(_os_shim.path.abspath(__file__)), "..", "common"))

import os
import numpy as np
import torch
import torch.nn as nn
import h5py

from ViT_rectangle import *
from ViT_sampling_rectangle_GPU import *

# ── Paths ─────────────────────────────────────────────────────────────────────
parent_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
checkpoint_dir = os.path.join(parent_dir, "checkpoints_fluvial_64")
data_dir       = os.path.join(parent_dir, "data")
data_path      = os.path.join(data_dir, "fluvial_binary_64x64.h5")
out_path       = os.path.join(data_dir, "conditional_samples_fluvial_64.npz")

# ── Config ────────────────────────────────────────────────────────────────────
N_SAMPLES     = 1000
BATCH_SIZE    = 100
PATCH_SIZE    = 4
IMAGE_SIZE    = (64, 64)   # (H, W)
OBSERVED_COLS = [22, 43]    # pixel columns to condition on (64 pixels each)

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Load test image ───────────────────────────────────────────────────────────
with h5py.File(data_path, "r") as f:
    imgs = f["images"][:]

test_image = imgs[-1000]           # first image of the held-out test set
H, W       = test_image.shape
print(f"Test image shape: H={H}, W={W}")
assert (H, W) == IMAGE_SIZE, f"Expected {IMAGE_SIZE}, got ({H}, {W})"

# ── Build condition_indices / condition_values ────────────────────────────────
# Observe every pixel in the two specified columns (H=48 pixels each).
cond_indices_list = []
cond_values_list  = []
for col in OBSERVED_COLS:
    for row in range(H):
        cond_indices_list.append(row * W + col)
        cond_values_list.append(int(test_image[row, col]))

condition_indices = np.array(cond_indices_list, dtype=np.int64)
condition_values  = np.array(cond_values_list,  dtype=np.int64)
print(f"Conditioning on {len(condition_indices)} pixels "
      f"({H} per column, columns {OBSERVED_COLS})")

# ── Load vocab + latest model checkpoint ─────────────────────────────────────
vocab_ckpt = torch.load(os.path.join(checkpoint_dir, "vocab_checkpoint.pt"), map_location="cpu")
vocab      = vocab_ckpt["vocab"]
mask_token = nn.Parameter(vocab_ckpt["mask_token"], requires_grad=True)

ckpt_files = sorted(
    [f for f in os.listdir(checkpoint_dir) if f.startswith("model_epoch_") and f.endswith(".pth")],
    key=lambda f: int(f.split("_")[-1].replace(".pth", ""))
)
if not ckpt_files:
    raise FileNotFoundError(f"No model_epoch_*.pth found in {checkpoint_dir}")
latest_ckpt = os.path.join(checkpoint_dir, ckpt_files[-1])
print(f"Loading checkpoint: {latest_ckpt}")
state = torch.load(latest_ckpt, map_location="cpu")

# ── Build model on CPU first ──────────────────────────────────────────────────
grid_h      = H // PATCH_SIZE
grid_w      = W // PATCH_SIZE
num_patches = grid_h * grid_w
patch_dim   = PATCH_SIZE * PATCH_SIZE
emb_dim     = state["patch_proj.weight"].shape[0]
num_layers  = sum(1 for k in state if k.startswith("layers.") and k.endswith(".norm1.weight"))

model = StackedContextViT(
    vocab        = vocab,
    mask_token   = mask_token,
    patch_dim    = patch_dim,
    num_patches  = num_patches,
    grid_size    = (grid_h, grid_w),
    emb_dim      = emb_dim,
    num_heads    = 2,
    num_layers   = num_layers,
    ffn_dim      = 256,
    use_pos_emb  = True,
    use_rel_bias = True,
)

# Load weights - allow mask_token to be missing if state dict doesn't have it
missing, unexpected = model.load_state_dict(state, strict=False)
if missing:
    print(f"Missing keys (using init values): {missing}")
if unexpected:
    print(f"Unexpected keys (ignored): {unexpected}")

model.to(device)
model.eval()
print("Model loaded.")

# ── Generate ──────────────────────────────────────────────────────────────────
all_images     = []
all_log_cond   = []
all_log_uncond = []
n_batches      = N_SAMPLES // BATCH_SIZE

for i in range(n_batches):
    print(f"Batch {i+1}/{n_batches} ...", flush=True)
    imgs_batch, lc, lu = generate_images_batch(
        model             = model,
        patch_size        = PATCH_SIZE,
        image_size        = IMAGE_SIZE,
        batch_size        = BATCH_SIZE,
        device            = device,
        condition_indices = condition_indices,
        condition_values  = condition_values,
        generation_order  = "manhattan",
        temperature       = 1.0,
    )
    all_images.append(imgs_batch.cpu().numpy())
    all_log_cond.append(lc.cpu().numpy())
    all_log_uncond.append(lu.cpu().numpy())

all_images     = np.concatenate(all_images,     axis=0).astype(np.float32)
all_log_cond   = np.concatenate(all_log_cond,   axis=0).astype(np.float32)
all_log_uncond = np.concatenate(all_log_uncond, axis=0).astype(np.float32)

print(f"Generated {len(all_images)} images, shape {all_images.shape[1:]}")
print(f"log_lik_cond   range: [{all_log_cond.min():.2f},  {all_log_cond.max():.2f}]")
print(f"log_lik_uncond range: [{all_log_uncond.min():.2f}, {all_log_uncond.max():.2f}]")

np.savez_compressed(
    out_path,
    images         = all_images,
    log_lik_cond   = all_log_cond,
    log_lik_uncond = all_log_uncond,
    observed_image = test_image.astype(np.float32),
    condition_cols = np.array(OBSERVED_COLS, dtype=np.int32),
)
print(f"Saved to {out_path}")

# ── Importance-weight ESS sanity check ───────────────────────────────────────
log_w  = all_log_uncond - all_log_cond
log_w -= log_w.max()
w      = np.exp(log_w)
w     /= w.sum()
ess    = 1.0 / (w ** 2).sum()
print(f"Importance weight ESS: {ess:.1f} / {N_SAMPLES}")

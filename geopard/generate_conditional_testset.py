"""
generate_conditional_testset.py

Runs the conditional inference experiment over all 1000 held-out test images.
For each image, observes all pixels in OBSERVED_COLS, draws N_SAMPLES posterior
samples, and records the unweighted mean, IS-weighted mean, and ESS.

Set SMOKE_TEST=1 to run on only 10 images for timing purposes.

Output:  data/fluvial64_testset_experiment.npz
    observed_images   float32  [N_TEST, H, W]  – the test images
    mean_unweighted   float32  [N_TEST, H, W]  – unweighted sample mean
    mean_weighted     float32  [N_TEST, H, W]  – IS-weighted posterior mean
    ess               float32  [N_TEST]         – effective sample size
    condition_cols    int32    [C]              – column indices observed
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

# ── Config ────────────────────────────────────────────────────────────────────
N_SAMPLES     = 5000
BATCH_SIZE    = 1000
PATCH_SIZE    = 4
IMAGE_SIZE    = (64, 64)
OBSERVED_COLS = [22, 43]
N_TEST        = 1000

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
if SMOKE_TEST:
    N_TEST   = 10
    out_path = os.path.join(data_dir, "fluvial64_testset_experiment_smoke.npz")
    print("SMOKE TEST: running on 10 images only.")
else:
    out_path = os.path.join(data_dir, "fluvial64_testset_experiment.npz")

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Load test images ──────────────────────────────────────────────────────────
with h5py.File(data_path, "r") as f:
    all_imgs = f["images"][:]

test_images = all_imgs[-1000:][:N_TEST]   # always take from the end; smoke grabs first 10
H, W        = test_images.shape[1:]
assert (H, W) == IMAGE_SIZE, f"Expected {IMAGE_SIZE}, got ({H}, {W})"
print(f"Loaded {N_TEST} test images, shape ({H}, {W})")

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

# ── Build model ───────────────────────────────────────────────────────────────
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

missing, unexpected = model.load_state_dict(state, strict=False)
if missing:
    print(f"Missing keys (using init values): {missing}")
if unexpected:
    print(f"Unexpected keys (ignored): {unexpected}")

model.to(device)
model.eval()
print("Model loaded.")

# ── Output buffers ────────────────────────────────────────────────────────────
out_observed = np.empty((N_TEST, H, W), dtype=np.float32)
out_mean_unw = np.empty((N_TEST, H, W), dtype=np.float32)
out_mean_w   = np.empty((N_TEST, H, W), dtype=np.float32)
out_ess      = np.empty(N_TEST,          dtype=np.float32)

n_batches = N_SAMPLES // BATCH_SIZE

# ── Main loop ─────────────────────────────────────────────────────────────────
import time
t_start = time.time()

for img_idx in range(N_TEST):
    test_image = test_images[img_idx]

    cond_indices_list, cond_values_list = [], []
    for col in OBSERVED_COLS:
        for row in range(H):
            cond_indices_list.append(row * W + col)
            cond_values_list.append(int(test_image[row, col]))
    condition_indices = np.array(cond_indices_list, dtype=np.int64)
    condition_values  = np.array(cond_values_list,  dtype=np.int64)

    imgs_acc       = []
    log_cond_acc   = []
    log_uncond_acc = []

    for b in range(n_batches):
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
        imgs_acc.append(imgs_batch.cpu().numpy())
        log_cond_acc.append(lc.cpu().numpy())
        log_uncond_acc.append(lu.cpu().numpy())

    imgs_np    = np.concatenate(imgs_acc,       axis=0).astype(np.float32)
    log_cond   = np.concatenate(log_cond_acc,   axis=0).astype(np.float32)
    log_uncond = np.concatenate(log_uncond_acc, axis=0).astype(np.float32)

    log_w  = log_uncond - log_cond
    log_w -= log_w.max()
    w      = np.exp(log_w)
    w     /= w.sum()
    ess    = 1.0 / (w ** 2).sum()

    out_observed[img_idx] = test_image.astype(np.float32)
    out_mean_unw[img_idx] = imgs_np.mean(axis=0)
    out_mean_w[img_idx]   = (w[:, None, None] * imgs_np).sum(axis=0)
    out_ess[img_idx]      = ess
    
    if (img_idx + 1) % 50 == 0:
        np.savez_compressed(
            out_path.replace(".npz", f"_checkpoint_{img_idx+1}.npz"),
            observed_images = out_observed[:img_idx+1],
            mean_unweighted = out_mean_unw[:img_idx+1],
            mean_weighted   = out_mean_w[:img_idx+1],
            ess             = out_ess[:img_idx+1],
            condition_cols  = np.array(OBSERVED_COLS, dtype=np.int32),
        )
        print(f"  → Checkpoint saved at image {img_idx+1}", flush=True)
    
    elapsed = time.time() - t_start
    elapsed = time.time() - t_start
    rate    = (img_idx + 1) / elapsed
    eta     = (N_TEST - img_idx - 1) / rate / 3600
    print(f"[{img_idx+1:4d}/{N_TEST}]  ESS={ess:.1f} ({100*ess/N_SAMPLES:.1f}%)  "
          f"elapsed={elapsed/60:.1f}m  ETA={eta:.2f}h", flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
np.savez_compressed(
    out_path,
    observed_images = out_observed,
    mean_unweighted = out_mean_unw,
    mean_weighted   = out_mean_w,
    ess             = out_ess,
    condition_cols  = np.array(OBSERVED_COLS, dtype=np.int32),
)
print(f"\nSaved to {out_path}")
print(f"ESS  mean={out_ess.mean():.1f}  min={out_ess.min():.1f}  max={out_ess.max():.1f}")
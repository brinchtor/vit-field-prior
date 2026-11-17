"""
generate_rowonly_bigger_random.py

Generates N_SAMPLES unconditional samples from one checkpoint of the larger
row-only model, sampling in random patch order. Saved to
data/generated_rowonly_bigger_random/.

Usage:
    python generate_rowonly_bigger_random.py
"""

import os
import re
import numpy as np
import torch
import torch.nn as nn

from ViT_rowonly import *
from ViT_sampling_rowonly import *

# ── Paths ─────────────────────────────────────────────────────────────────────
parent_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
checkpoint_dir = os.path.join(parent_dir, "checkpoints_fluvial_64_rowonly_bigger")
data_dir       = os.path.join(parent_dir, "data")
out_dir        = os.path.join(data_dir, "generated_rowonly_bigger_random")
os.makedirs(out_dir, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
EPOCH            = None          # None = latest checkpoint in the folder
GENERATION_ORDER = "random"
TAG              = "rowonly_bigger_random"
N_SAMPLES        = 1000
BATCH_SIZE       = 100
PATCH_SIZE       = 4
IMAGE_SIZE       = (64, 64)      # (H, W)
NUM_HEADS        = 4
FFN_DIM          = 512
H, W = IMAGE_SIZE

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Pick checkpoint ───────────────────────────────────────────────────────────
pattern = re.compile(r"^model_epoch_(\d+)\.pth$")
epochs  = sorted(int(pattern.match(f).group(1)) for f in os.listdir(checkpoint_dir) if pattern.match(f))
if not epochs:
    raise FileNotFoundError(f"No model_epoch_*.pth found in {checkpoint_dir}")
epoch = EPOCH if EPOCH is not None else epochs[-1]
print(f"Using checkpoint epoch {epoch} (available: {epochs})")

out_path = os.path.join(out_dir, f"{TAG}_epoch_{epoch:04d}_n{N_SAMPLES}.npz")

# ── Load vocab + checkpoint ───────────────────────────────────────────────────
vocab_ckpt = torch.load(os.path.join(checkpoint_dir, "vocab_checkpoint.pt"), map_location="cpu")
vocab      = vocab_ckpt["vocab"]
mask_token = nn.Parameter(vocab_ckpt["mask_token"], requires_grad=True)
state      = torch.load(os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth"), map_location="cpu")

grid_h      = H // PATCH_SIZE
grid_w      = W // PATCH_SIZE
num_patches = grid_h * grid_w
patch_dim   = PATCH_SIZE * PATCH_SIZE
emb_dim     = state["patch_proj.weight"].shape[0]
num_layers  = sum(1 for k in state if k.startswith("layers.") and k.endswith(".norm1.weight"))
assert state["pos_emb"].shape[0] == grid_h, "expects a row-only checkpoint"
assert state["layers.0.ffn.0.weight"].shape[0] == FFN_DIM, "FFN_DIM does not match checkpoint"

model = StackedContextViT(
    vocab        = vocab,
    mask_token   = mask_token,
    patch_dim    = patch_dim,
    num_patches  = num_patches,
    grid_size    = (grid_h, grid_w),
    emb_dim      = emb_dim,
    num_heads    = NUM_HEADS,
    num_layers   = num_layers,
    ffn_dim      = FFN_DIM,
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

# ── Generate ──────────────────────────────────────────────────────────────────
all_images = []
n_batches  = N_SAMPLES // BATCH_SIZE
for i in range(n_batches):
    print(f"Batch {i+1}/{n_batches} ...", flush=True)
    imgs, _, _ = generate_images_batch(
        model            = model,
        patch_size       = PATCH_SIZE,
        image_size       = IMAGE_SIZE,
        batch_size       = BATCH_SIZE,
        device           = device,
        generation_order = GENERATION_ORDER,
        temperature      = 1.0,
    )
    all_images.append(imgs.cpu().numpy())

all_images = np.concatenate(all_images, axis=0).astype(np.float32)
print(f"Generated {all_images.shape[0]} images, shape {all_images.shape[1:]}")
np.savez_compressed(out_path, images=all_images)
print(f"Saved to {out_path}")

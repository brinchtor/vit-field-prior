"""
generate_all_checkpoints_rowonly_bigger.py

Generates N_SAMPLES unconditional samples from every model_epoch_*.pth found in
checkpoints_fluvial_64_rowonly_bigger (larger row-only model: 4 heads, ffn 512).
One .npz per checkpoint, saved in a new folder data/generated_rowonly_bigger/.
Existing outputs are skipped, so the script can be rerun.

Usage:
    python generate_all_checkpoints_rowonly_bigger.py
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
out_dir        = os.path.join(data_dir, "generated_rowonly_bigger")
os.makedirs(out_dir, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
TAG           = "rowonly_bigger"
NUM_HEADS     = 4
FFN_DIM       = 512
N_SAMPLES     = 1000
BATCH_SIZE    = 100
PATCH_SIZE    = 4
IMAGE_SIZE    = (64, 64)   # (H, W)
H = IMAGE_SIZE[0]
W = IMAGE_SIZE[1]

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Load vocab ────────────────────────────────────────────────────────────────
vocab_ckpt = torch.load(os.path.join(checkpoint_dir, "vocab_checkpoint.pt"), map_location="cpu")
vocab      = vocab_ckpt["vocab"]

# ── Find all model checkpoints, sorted by epoch ───────────────────────────────
pattern    = re.compile(r"^model_epoch_(\d+)\.pth$")
ckpt_files = sorted(
    [f for f in os.listdir(checkpoint_dir) if pattern.match(f)],
    key=lambda f: int(pattern.match(f).group(1)),
)
if not ckpt_files:
    raise FileNotFoundError(f"No model_epoch_*.pth found in {checkpoint_dir}")
print(f"Found {len(ckpt_files)} checkpoints in {checkpoint_dir}")

grid_h      = H // PATCH_SIZE
grid_w      = W // PATCH_SIZE
num_patches = grid_h * grid_w
patch_dim   = PATCH_SIZE * PATCH_SIZE
n_batches   = N_SAMPLES // BATCH_SIZE

# ── Loop over checkpoints ─────────────────────────────────────────────────────
for ckpt_name in ckpt_files:
    epoch    = int(pattern.match(ckpt_name).group(1))
    out_path = os.path.join(out_dir, f"{TAG}_epoch_{epoch:04d}_n{N_SAMPLES}.npz")

    if os.path.exists(out_path):
        print(f"[epoch {epoch}] {out_path} exists, skipping")
        continue

    print(f"\n[epoch {epoch}] Loading {ckpt_name}")
    state = torch.load(os.path.join(checkpoint_dir, ckpt_name), map_location="cpu")

    mask_token = nn.Parameter(vocab_ckpt["mask_token"], requires_grad=True)
    emb_dim    = state["patch_proj.weight"].shape[0]
    assert emb_dim % NUM_HEADS == 0, f"emb_dim {emb_dim} not divisible by NUM_HEADS {NUM_HEADS}"
    assert state["layers.0.ffn.0.weight"].shape[0] == FFN_DIM, (
        f"checkpoint ffn_dim {state['layers.0.ffn.0.weight'].shape[0]} != FFN_DIM {FFN_DIM}")
    num_layers = sum(1 for k in state if k.startswith("layers.") and k.endswith(".norm1.weight"))
    assert state["pos_emb"].shape[0] == grid_h, (
        f"pos_emb has {state['pos_emb'].shape[0]} rows, expected grid_h={grid_h} (row-only checkpoint)")

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

    all_images = []
    for i in range(n_batches):
        print(f"[epoch {epoch}] Batch {i+1}/{n_batches} ...", flush=True)
        imgs, _, _ = generate_images_batch(
            model            = model,
            patch_size       = PATCH_SIZE,
            image_size       = IMAGE_SIZE,
            batch_size       = BATCH_SIZE,
            device           = device,
            generation_order = "raster",
            temperature      = 1.0,
        )
        all_images.append(imgs.cpu().numpy())

    all_images = np.concatenate(all_images, axis=0).astype(np.float32)
    print(f"[epoch {epoch}] Generated {all_images.shape[0]} images, shape {all_images.shape[1:]}")
    np.savez_compressed(out_path, images=all_images)
    print(f"[epoch {epoch}] Saved to {out_path}")

    del model, state, all_images
    if device.type == "cuda":
        torch.cuda.empty_cache()

print("\nDone.")

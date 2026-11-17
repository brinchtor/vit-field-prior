"""
conditional_model_gauss_C_wells.py

Gaussian-observation ViT proposal for model C, wells regime. Loads the noisy
observation dataset, draws S proposal samples per image (each first samples the
observed pixels from p(x_i | y_i) using the prior marginal, then completes the
field with the masked ViT), and saves samples plus the three log-quantities
needed for the IS weight.

Output: data/gauss_exp/vit_C_wells.npz
    samples           float32  [N, S, H, W]
    log_lik_cond      float32  [N, S]   -- log q(x | x_K)
    log_lik_uncond    float32  [N, S]   -- log p_theta(x)
    log_marginal_sum  float32  [N, S]   -- sum_i log m_i(x_i)

Per-image IS log-weight (per-observation constant cancels under self-normalization):
    log w = log_lik_uncond - log_lik_cond - log_marginal_sum
"""
import os as _os_shim, sys as _sys_shim
_sys_shim.path.insert(0, _os_shim.path.join(_os_shim.path.dirname(_os_shim.path.abspath(__file__)), "..", "common"))

import os
import numpy as np
import torch
import torch.nn as nn

from ViT_rectangle import StackedContextViT
from ViT_sampling_rectangle_GPU import generate_images_batch_gauss

# -- Config ---------------------------------------------------------------------
parent_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir       = os.path.join(parent_dir, "data")
gauss_dir      = os.path.join(data_dir, "gauss_exp")

DATA_PATH      = os.path.join(gauss_dir, "obs_C_wells.npz")
MARGINALS_PATH = os.path.join(data_dir, "marginals.npz")
CHECKPOINT_DIR = os.path.join(parent_dir, "checkpoints_MRF_2ndorder_big")
OUT_PATH       = os.path.join(gauss_dir, "vit_C_wells.npz")

MARGINAL_KEY   = "C"
SIGMA          = 0.5

S              = 1000        # posterior samples per image
BATCH_SIZE     = 100
PATCH_SIZE     = 2
IMAGE_SIZE     = (12, 24)   # (H, W) -- match your MRF grid
H = IMAGE_SIZE[0]
W = IMAGE_SIZE[1]

# -- Device ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -- Load vocab + latest model checkpoint --------------------------------------
vocab_ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "vocab_checkpoint.pt"), map_location="cpu")
vocab      = vocab_ckpt["vocab"]
mask_token = nn.Parameter(vocab_ckpt["mask_token"], requires_grad=True)

ckpt_files = sorted(
    [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("model_epoch_") and f.endswith(".pth")],
    key=lambda f: int(f.split("_")[-1].replace(".pth", ""))
)
if not ckpt_files:
    raise FileNotFoundError(f"No model_epoch_*.pth found in {CHECKPOINT_DIR}")
latest_ckpt = os.path.join(CHECKPOINT_DIR, ckpt_files[-1])
print(f"Loading checkpoint: {latest_ckpt}")
state = torch.load(latest_ckpt, map_location="cpu")

# -- Build model on CPU first ---------------------------------------------------
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

# -- Load prior marginals (P(x_i = 1) field for this model) ---------------------
marginal = np.load(MARGINALS_PATH)[MARGINAL_KEY]   # [H, W, 2]
assert marginal.shape == (H, W, 2), f"marginal shape {marginal.shape} != {(H, W, 2)}"

# -- Load dataset ---------------------------------------------------------------
data        = np.load(DATA_PATH)
vit_indices = data["vit_indices"]   # (N, K) int
vit_values  = data["vit_values"]    # (N, K) float (noisy y)
N           = vit_indices.shape[0]

# -- Sample ---------------------------------------------------------------------
all_samples  = np.empty((N, S, H, W), dtype=np.float32)
all_log_cond = np.empty((N, S),       dtype=np.float32)
all_log_unc  = np.empty((N, S),       dtype=np.float32)
all_log_marg = np.empty((N, S),       dtype=np.float32)

for i in range(N):
    print(f"Image {i+1}/{N} ...", flush=True)
    imgs, lc, lu, lm = generate_images_batch_gauss(
        model             = model,
        patch_size        = PATCH_SIZE,
        image_size        = IMAGE_SIZE,
        batch_size        = S,
        device            = device,
        condition_indices = vit_indices[i],
        condition_y       = vit_values[i],
        marginal          = marginal,
        sigma             = SIGMA,
        generation_order  = "manhattan",
        temperature       = 1.0,
    )
    all_samples[i]  = imgs.cpu().numpy()
    all_log_cond[i] = lc.cpu().numpy()
    all_log_unc[i]  = lu.cpu().numpy()
    all_log_marg[i] = lm.cpu().numpy()

# -- Save -----------------------------------------------------------------------
np.savez_compressed(
    OUT_PATH,
    samples          = all_samples,
    log_lik_cond     = all_log_cond,
    log_lik_uncond   = all_log_unc,
    log_marginal_sum = all_log_marg,
)
print(f"Saved to {OUT_PATH}")

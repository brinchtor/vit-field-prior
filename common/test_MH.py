# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:16:19 2026

@author: brinc
"""

import numpy as np
import torch
import torch.nn as nn

from getQ import get_q
from ViT_sampling_rectangle_debug import *
from ViT_rectangle import *
from MH_ViT import *
from test_functions import *

MRF_NPZ  = "data/mrf_dataset_march.npz"
IMG_KEY  = "images"

VIT_CKPT = "outputs_0303/vocab_checkpoint.pt"
PATCH_SIZE = 2
GEN_ORDER = "raster"
BATCH_SIZE = 32

OUT_NPZ = "logliks_out_4.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =============================================

mrf_imgs = (np.load(MRF_NPZ)[IMG_KEY].astype(np.int64))[:1000]

train_tensor = torch.tensor(mrf_imgs, dtype=torch.float32)
checkpoint = torch.load(VIT_CKPT, map_location="cpu")

vocab = checkpoint["vocab"]
counts = checkpoint["counts"]

mask_token = nn.Parameter(checkpoint["mask_token"], requires_grad=True)

patch_size = checkpoint["patch_size"]
cap_size = checkpoint["cap_size"]

nx, ny = train_tensor.shape[1:]
num_patches = (nx // patch_size) * (ny // patch_size)
patch_dim   = patch_size * patch_size

model = StackedContextViT(
    vocab=vocab,
    mask_token=mask_token,
    patch_dim=patch_dim,
    num_patches=num_patches,
    grid_size=(6, 12),
    emb_dim=64,
    num_heads=2,
    num_layers=2,
    ffn_dim=256,
    use_pos_emb=True,
    pos_emb_init=None,
    use_rel_bias=True,
    rel_bias_init=None
).to(DEVICE)

model.load_state_dict(torch.load("outputs_0303/model_epoch_1000.pth", map_location="cpu"))

image = mrf_imgs[1]  # shape (H, W)

# --- Line observations ---
test_exact(image, model)
test_gauss(image, model, sigma=0.3)
test_sum(image, model)
# --- Random observations ---
test_exact_random(image, model, n_obs=20)
test_gauss_random(image, model, n_obs=20, sigma=0.3)
test_sum_random(image, model, n_obs=20)
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 12:20:56 2025
@author: brinc
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from tqdm.notebook import tqdm
import copy
from ViT_rowonly import *
from ViT_sampling_rowonly import *
from training_rowonly import *
from tqdm import tqdm  # For progress bars
import time
import io
import h5py

# ── Paths ────────────────────────────────────────────────────────────────────
# All three sibling folders (code/, data/, outputs/) live under the same parent
parent_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path   = os.path.join(parent_dir, "data", "fluvial_binary_64x64.h5")
output_dir  = os.path.join(parent_dir, "checkpoints_fluvial_64_rowonly_bigger")
os.makedirs(output_dir, exist_ok=True)

# ── Load dataset ─────────────────────────────────────────────────────────────
with h5py.File(data_path, "r") as f:
    imgs = f["images"][:]          # uint8, shape (50000, 256, 256)

# Hold out the last 1000 images for testing
train_imgs = imgs[:-1000]          # (49000, 256, 256)

train_tensor = torch.tensor(train_imgs, dtype=torch.float32)
print(f'Dataset size (train): {len(train_tensor)}', flush=True)
print(f'Image size:           {train_tensor.shape[1:]}')

# ── Training ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, _ = run_training(
    training_data   = train_tensor,   # (N, H, W) float
    nr_epochs = 10 if os.environ.get("SMOKE_TEST") == "1" else 1000,
    batch_size      = 128,
    mask_rate       = 0.01,
    final_mask_rate = 0.99,
    mask_schedule   = "cyclic",
    patch_size      = 4,
    num_heads       = 4,
    num_layers      = 4,
    ffn_dim         = 512,
    learning_rate   = 1e-3,
    emb_dim         = 128,
    vocab_cap       = 6000,
    pretrained      = False,
    use_pos_emb     = True,
    pos_emb         = None,
    use_rel_bias    = True,
    rel_bias        = None,
    foldername      = output_dir,
)

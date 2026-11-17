# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 12:20:56 2025
@author: brinc
"""
import os as _os_shim, sys as _sys_shim
_sys_shim.path.insert(0, _os_shim.path.join(_os_shim.path.dirname(_os_shim.path.abspath(__file__)), "..", "common"))
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from tqdm.notebook import tqdm
import copy
from ViT_rectangle import *
from ViT_sampling_rectangle import *
from training_rectangle import *
from tqdm import tqdm  # For progress bars
import time
import io

# ── Paths ────────────────────────────────────────────────────────────────────
# All three sibling folders (code/, data/, outputs/) live under the same parent
parent_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path   = os.path.join(parent_dir, "data", "mrf_dataset_D.npz")
output_dir  = os.path.join(parent_dir, "checkpoints_MRF_model_D")

# ── Load dataset ─────────────────────────────────────────────────────────────

npz = np.load(data_path)
imgs = npz["images"]
train_imgs = torch.tensor(imgs, dtype=torch.float32)

train_tensor = torch.tensor(train_imgs, dtype=torch.float32)
print(f'Dataset size (train): {len(train_tensor)}', flush=True)
print(f'Image size:           {train_tensor.shape[1:]}')

# ── Training ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, _ = run_training(
    training_data   = train_tensor,   # (N, H, W) float
    nr_epochs = 1000,
    batch_size      = 100,
    mask_rate       = 0.01,
    final_mask_rate = 0.99,
    mask_schedule   = "cyclic",
    patch_size      = 2,
    num_heads       = 2,
    num_layers      = 2,
    ffn_dim         = 256,
    learning_rate   = 1e-3,
    emb_dim         = 64,
    vocab_cap       = 16,
    pretrained      = False,
    use_pos_emb     = True,
    pos_emb         = None,
    use_rel_bias    = True,
    rel_bias        = None,
    foldername      = output_dir,
)
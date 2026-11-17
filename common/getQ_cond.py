# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 09:50:29 2026

@author: brinc
"""

import numpy as np
from getQ import *
from MRF_functions import *
import torch

def get_condmap(img, patch_size, patch_mask):
    """
    img: (H,W) or (1,H,W) torch/numpy with values in {0,1} (or close).
    patch_mask: (N,) or (Gh,Gw) boolean. True=masked, False=observed (ViT uses mask_token there).
    returns: [((r,c), val), ...] for all pixels in observed (unmasked) patches.
    """
    # to numpy (H,W)
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]

    H, W = img.shape
    Gh, Gw = H // patch_size, W // patch_size
    N = Gh * Gw

    # flatten mask to (N,)
    if isinstance(patch_mask, torch.Tensor):
        patch_mask = patch_mask.detach().cpu().numpy()
    patch_mask = np.asarray(patch_mask).astype(bool)
    if patch_mask.ndim == 2:
        patch_mask = patch_mask.reshape(-1)
    if patch_mask.shape[0] != N:
        raise ValueError(f"patch_mask has length {patch_mask.shape[0]} but expected {N}")

    condmap = []
    for p in range(N):
        if patch_mask[p]:
            continue  # masked patch -> not observed
        pr, pc = divmod(p, Gw)
        r0, c0 = pr * patch_size, pc * patch_size
        patch = img[r0:r0+patch_size, c0:c0+patch_size]
        # add every pixel in this observed patch
        for lr in range(patch_size):
            for lc in range(patch_size):
                condmap.append(((r0+lr, c0+lc), int(np.rint(patch[lr, lc]))))

    return condmap

def sample_conditional(
    alf: np.ndarray,
    bet: float,
    N: int,
    condmap,
    seed: int = 0
):
    d = alf.shape[2]
    cond_alf = np.array(alf, dtype=float, copy=True)
    for pos, val in condmap:
        cond_alf[pos] = np.full(d, -np.inf)
        cond_alf[pos][val] = 0.0
    samples = generate_samples(cond_alf, bet, N, seed, force_n1_smaller=False)
    
    return samples

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 11:33:54 2026

@author: brinc

Anisotropic-diagonal version: uses separate gamma_ne and gamma_nw couplings.
"""
import os as _os_shim, sys as _sys_shim
_sys_shim.path.insert(0, _os_shim.path.join(_os_shim.path.dirname(_os_shim.path.abspath(__file__)), "..", "common"))

import numpy as np
import torch
import torch.nn as nn

from getQ_2ndorder_aniso import get_q
from ViT_sampling_rectangle_debug import *
from ViT_rectangle import *

# ================== EDIT ME ==================
VIT_NPZ  = "data/ViT-model-D.npz"
MRF_NPZ  = "data/mrf-model-D.npz"
IMG_KEY  = "images"                 # key inside all npz files

n1, n2, d = 12, 24, 2
ALF = np.zeros((n1, n2, d), dtype=float)
for i in range(n1):
    ALF[i, :, 0] = np.linspace(-0.25, 0.25, n2)
BET = 0.5
GAMMA_NE = 0.3
GAMMA_NW = 0.1

VIT_CKPT = "outputs-model-D/vocab_checkpoint.pt"
PATCH_SIZE = 2
GEN_ORDER = "raster"
BATCH_SIZE = 32

OUT_NPZ = "logliks-model-D.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =============================================


def mrf_log_unnormalized(x, alf, bet, gamma_ne, gamma_nw):
    """
    x: (B, n1, n2) integer labels in {0..d-1}
    alf: (n1, n2, d) unary log-potentials
    bet: scalar coupling for axis-aligned (horizontal + vertical) pairs
    gamma_ne: scalar coupling for the NE/SW diagonal
    gamma_nw: scalar coupling for the NW/SE diagonal

    Convention matches qeval in getQ_2ndorder_aniso.py:
        NE/SW diagonal couples (i, j) with (i-1, j+1)
            -> anti-diagonal in array indexing:
               x[:, :-1, 1:] vs x[:, 1:, :-1]
        NW/SE diagonal couples (i, j) with (i+1, j+1)
            -> main diagonal in array indexing:
               x[:, :-1, :-1] vs x[:, 1:, 1:]
    """
    x = np.asarray(x)
    n1, n2, d = alf.shape
    assert x.ndim == 3 and x.shape[1:] == (n1, n2)

    rr = np.arange(n1)[:, None]
    cc = np.arange(n2)[None, :]
    unary = alf[rr, cc, x].sum(axis=(1, 2))

    horiz = (x[:, :, :-1] == x[:, :, 1:]).sum(axis=(1, 2))
    vert  = (x[:, :-1, :] == x[:, 1:, :]).sum(axis=(1, 2))
    axis_aligned = bet * (horiz + vert)

    # NW/SE diagonal: (i,j) ~ (i+1,j+1)
    nw_diag = (x[:, :-1, :-1] == x[:, 1:, 1:]).sum(axis=(1, 2))
    # NE/SW diagonal: (i,j) ~ (i-1,j+1)  <=>  (i+1,j) ~ (i,j+1)
    ne_diag = (x[:, :-1, 1:]  == x[:, 1:, :-1]).sum(axis=(1, 2))

    diag = gamma_nw * nw_diag + gamma_ne * ne_diag

    return unary + axis_aligned + diag


def vit_loglik(model, imgs_int, patch_size, gen_order, batch_size, device):
    imgs = torch.tensor(imgs_int, dtype=torch.float32, device=device)  # (B,H,W)
    outs = []
    with torch.no_grad():
        for i in range(0, imgs.shape[0], batch_size):
            ll = log_likelihood_evaluation_batch(
                model=model,
                batch_images=imgs[i:i+batch_size],
                patch_size=patch_size,
                condition_indices=None,
                condition_values=None,
                generation_order=gen_order,
            )
            outs.append(ll.detach().cpu().numpy())
    return np.concatenate(outs, axis=0)

def vit_loglik_and_contribs(model, imgs_int, patch_size, gen_order, batch_size, device):
    imgs = torch.tensor(imgs_int, dtype=torch.float32, device=device)  # (B,H,W)

    ll_outs = []
    contrib_outs = []
    order_saved = None  # optional: save the reveal order once

    with torch.no_grad():
        for i in range(0, imgs.shape[0], batch_size):
            ll, contribs, dbg = log_likelihood_evaluation_batch_debug(
                model=model,
                batch_images=imgs[i:i+batch_size],
                patch_size=patch_size,
                condition_indices=None,
                condition_values=None,
                generation_order=gen_order,
                return_cpu=True,   # <-- if your debug func has this flag
            )

            # ll: [b], contribs: [b, T]
            ll_outs.append(ll.numpy())
            contrib_outs.append(contribs.numpy().astype(np.float32))  # float32 to shrink npz size

            if order_saved is None and dbg is not None and "order" in dbg:
                order_saved = np.asarray(dbg["order"], dtype=np.int32)

    ll_all = np.concatenate(ll_outs, axis=0)                  # [num_imgs]
    contrib_all = np.concatenate(contrib_outs, axis=0)        # [num_imgs, T]
    return ll_all, contrib_all, order_saved

def vit_loglik_contribs_trueids(model, imgs_int, patch_size, gen_order, batch_size, device):
    imgs = torch.tensor(imgs_int, dtype=torch.float32, device=device)  # (B,H,W)

    ll_outs = []
    contrib_outs = []
    trueid_outs = []
    order_saved = None

    with torch.no_grad():
        for i in range(0, imgs.shape[0], batch_size):
            ll, contribs, dbg = log_likelihood_evaluation_batch_debug(
                model=model,
                batch_images=imgs[i:i+batch_size],
                patch_size=patch_size,
                condition_indices=None,
                condition_values=None,
                generation_order=gen_order,
                return_cpu=True,   # remove if your debug func doesn't have this
            )

            ll_outs.append(ll.numpy())                                  # [b]
            contrib_outs.append(contribs.numpy().astype(np.float32))    # [b,T]

            # true ids: [b,T]
            true_ids = dbg["true_idx"]
            if torch.is_tensor(true_ids):
                true_ids = true_ids.cpu().numpy()
            trueid_outs.append(true_ids.astype(np.int16))               # small

            if order_saved is None and "order" in dbg:
                order_saved = np.asarray(dbg["order"], dtype=np.int32)

    ll_all = np.concatenate(ll_outs, axis=0)                 # [num_imgs]
    contrib_all = np.concatenate(contrib_outs, axis=0)       # [num_imgs,T]
    trueid_all = np.concatenate(trueid_outs, axis=0)         # [num_imgs,T]
    return ll_all, contrib_all, trueid_all, order_saved


def main():
    vit_imgs = np.load(VIT_NPZ)[IMG_KEY].astype(np.int64)

    # Keep MRF set aligned to ViT set length, as before
    mrf_imgs = (np.load(MRF_NPZ)[IMG_KEY].astype(np.int64))[:len(vit_imgs)]

    alf = ALF

    # exact logZ (one per alf, bet, gamma_ne, gamma_nw)
    _, logZ = get_q(alf, BET, GAMMA_NE, GAMMA_NW, return_logZ=True)
    logZ = float(logZ)

    # exact MRF loglik (under the exact MRF)
    vit_under_exact  = mrf_log_unnormalized(vit_imgs,  alf, BET, GAMMA_NE, GAMMA_NW) - logZ
    mrf_under_exact  = mrf_log_unnormalized(mrf_imgs,  alf, BET, GAMMA_NE, GAMMA_NW) - logZ

    # ViT-defined loglik
    train_tensor = torch.tensor(mrf_imgs, dtype=torch.float32)
    checkpoint = torch.load(VIT_CKPT, map_location="cpu")

    vocab = checkpoint["vocab"]
    counts = checkpoint["counts"]

    # rewrap mask token
    mask_token = torch.tensor([0.5000, 0.5000, 0.5000, 0.5000], requires_grad=True)

    patch_size = checkpoint["patch_size"]
    cap_size = checkpoint["cap_size"]

    nx, ny = train_tensor.shape[1:]
    num_patches = (nx // patch_size) * (ny // patch_size)
    patch_dim   = patch_size * patch_size

    # Recreate model with same dims as before
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
    
    
    model.load_state_dict(torch.load("outputs-model-D/model_epoch_1000.pth", map_location="cpu"))

    # (Optional but typically needed) load model weights if they exist in checkpoint
    # If your checkpoint already includes a model_state_dict, uncomment this:
    # if "model_state_dict" in checkpoint:
    #     model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    vit_under_vit, vit_contribs, vit_trueids, reveal_order = vit_loglik_contribs_trueids(
        model, vit_imgs, PATCH_SIZE, GEN_ORDER, BATCH_SIZE, DEVICE
    )
    mrf_under_vit, mrf_contribs, mrf_trueids, _ = vit_loglik_contribs_trueids(
        model, mrf_imgs, PATCH_SIZE, GEN_ORDER, BATCH_SIZE, DEVICE
    )
    
    vit_logp_flat   = vit_contribs.reshape(-1)
    vit_trueid_flat = vit_trueids.reshape(-1)
    
    mrf_logp_flat   = mrf_contribs.reshape(-1)
    mrf_trueid_flat = mrf_trueids.reshape(-1)
    
    vocab_bits = model.vocab.detach().cpu().numpy().astype(np.int8)  # [16,4] for 2x2
    vocab_str = np.array(
        ["(" + ",".join(map(str, row.tolist())) + ")" for row in vocab_bits],
        dtype=object
    )
    
    np.savez(
        OUT_NPZ,
        vit_under_exact=vit_under_exact,
        mrf_under_exact=mrf_under_exact,
        vit_under_vit=vit_under_vit,
        mrf_under_vit=mrf_under_vit,
        vit_contribs=vit_contribs,
        mrf_contribs=mrf_contribs,
        logZ=logZ,
        bet=BET,
        gamma_ne=GAMMA_NE,
        gamma_nw=GAMMA_NW,
        
        vit_trueids=vit_trueids,
        vit_logp_flat=vit_logp_flat,
        vit_trueid_flat=vit_trueid_flat,
    
        mrf_trueids=mrf_trueids,
        mrf_logp_flat=mrf_logp_flat,
        mrf_trueid_flat=mrf_trueid_flat,
        
        vocab_bits=vocab_bits,
        vocab_str=vocab_str
    )
    print("saved:", OUT_NPZ)


if __name__ == "__main__":
    main()
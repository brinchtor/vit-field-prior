# -*- coding: utf-8 -*-
"""
plot_diff_by_epoch.py

For every generated file in data/generated_old and data/generated_new, computes
(ViT prior mean - test set mean) and plots the difference maps, one figure per
model with one subplot per epoch (5 columns).

Layout (run from the code/ folder; code/, data/ and figures/ share one root):
    root/code/plot_diff_by_epoch.py
    root/data/fluvial_binary_64x64.h5
    root/data/fluvial_binary_128x64.h5
    root/data/generated_old/old_epoch_XXXX_n1000.npz
    root/data/generated_new/new_epoch_XXXX_n1000.npz
    root/figures/diff_by_epoch/   (created)

Usage:
    python plot_diff_by_epoch.py
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import h5py

# ── Paths ─────────────────────────────────────────────────────────────────────
try:
    code_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # e.g. running a cell in Spyder
    code_dir = os.getcwd()
root     = os.path.dirname(code_dir)
data_dir = os.path.join(root, "data")
out_dir  = os.path.join(root, "figures", "diff_by_epoch")
os.makedirs(out_dir, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
N_COLS      = 5
SHARED_SCALE = True    # one symmetric color range for all subplots of both figures
DPI         = 300
CROP_SEED   = 0        # seed for the random crop offsets of the test set
N_CROPS     = 1        # random crops per test image (1 = one crop per image)

# ── Test set means ────────────────────────────────────────────────────────────
with h5py.File(os.path.join(data_dir, "fluvial_binary_64x64.h5"), "r") as f:
    imgs_true = f["images"][-1000:]
mean_true_old = np.mean(imgs_true, axis=0)

with h5py.File(os.path.join(data_dir, "fluvial_binary_128x64.h5"), "r") as f:
    imgs_true = f["images"][-1000:]
crop_w  = 64
max_off = imgs_true.shape[2] - crop_w
rng     = np.random.default_rng(CROP_SEED)
crops   = []
for _ in range(N_CROPS):
    offs = rng.integers(0, max_off + 1, size=imgs_true.shape[0])   # uniform over all valid offsets
    crops.append(np.stack([img[:, o:o + crop_w] for img, o in zip(imgs_true, offs)]))
mean_true_new = np.mean(np.concatenate(crops, axis=0), axis=0)    # mean over random 64x64 crops

# ── Helpers ───────────────────────────────────────────────────────────────────
pattern = re.compile(r"_epoch_(\d+)_")

def list_generated(folder):
    """Return [(epoch, path), ...] sorted by epoch."""
    files = []
    for f in os.listdir(folder):
        m = pattern.search(f)
        if m and f.endswith(".npz"):
            files.append((int(m.group(1)), os.path.join(folder, f)))
    if not files:
        raise FileNotFoundError(f"No generated .npz files in {folder}")
    return sorted(files)

def compute_diffs(folder, mean_true):
    epochs, diffs = [], []
    for epoch, path in list_generated(folder):
        imgs = np.load(path)["images"]
        mean = np.mean(imgs, axis=0)
        if mean.shape != mean_true.shape:
            raise ValueError(f"{path}: mean shape {mean.shape} != test mean shape {mean_true.shape}")
        epochs.append(epoch)
        diffs.append(mean - mean_true)
        print(f"{os.path.basename(path)}: mean |diff| = {np.abs(diffs[-1]).mean():.4f}")
    return epochs, diffs

def diff_figure(epochs, diffs, title, savename, vmax):
    n      = len(diffs)
    ncols  = min(N_COLS, n)
    nrows  = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols + 1, 3.2 * nrows + 0.8),
                             constrained_layout=True, squeeze=False)
    im = None
    for ax in axes.ravel():
        ax.axis("off")
    for ax, epoch, diff in zip(axes.ravel(), epochs, diffs):
        v = vmax if vmax is not None else np.abs(diff).max()
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-v, vmax=v, aspect="equal")
        ax.set_title(f"Epoch {epoch}\nmean |diff| = {np.abs(diff).mean():.4f}", fontsize=10)
        ax.axis("off")
    if vmax is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="ViT prior mean − test set mean")
    fig.suptitle(title)
    fig.savefig(savename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {savename}")

# ── Compute and plot ──────────────────────────────────────────────────────────
epochs_old, diffs_old = compute_diffs(os.path.join(data_dir, "generated_old"), mean_true_old)
epochs_new, diffs_new = compute_diffs(os.path.join(data_dir, "generated_new"), mean_true_new)

vmax = None
if SHARED_SCALE:
    vmax = max(max(np.abs(d).max() for d in diffs_old),
               max(np.abs(d).max() for d in diffs_new))

diff_figure(epochs_old, diffs_old, "Old method, no crop: ViT prior mean − test set mean",
            os.path.join(out_dir, "diff_by_epoch_old.png"), vmax)
diff_figure(epochs_new, diffs_new, "New method, random crop: ViT prior mean − test set mean",
            os.path.join(out_dir, "diff_by_epoch_new.png"), vmax)

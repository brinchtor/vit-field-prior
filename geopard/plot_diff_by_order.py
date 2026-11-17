# -*- coding: utf-8 -*-
"""
plot_diff_by_order.py

Plots (ViT prior mean - test set mean) for the larger row-only model sampled in
three generation orders (raster, reverse raster, random), side by side with one
common colorbar. Run from anywhere; code/, data/ and figures/ share one root.

Reads:
    root/data/fluvial_binary_64x64.h5
    root/data/generated_rowonly_bigger/rowonly_bigger_epoch_XXXX_n1000.npz
    root/data/generated_rowonly_bigger_revraster/rowonly_bigger_revraster_epoch_XXXX_n1000.npz
    root/data/generated_rowonly_bigger_random/rowonly_bigger_random_epoch_XXXX_n1000.npz
Writes:
    root/figures/diff_by_order/diff_by_order_epoch_XXXX.png
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

# ── Paths ─────────────────────────────────────────────────────────────────────
try:
    code_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # e.g. running a cell in Spyder
    code_dir = os.getcwd()
root     = os.path.dirname(code_dir)
data_dir = os.path.join(root, "data")
out_dir  = os.path.join(root, "figures", "diff_by_order")
os.makedirs(out_dir, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
EPOCH = None     # None = latest epoch that exists for all three orders
DPI   = 300
ORDERS = [       # (panel title, data subfolder, file tag)
    ("Raster",         "generated_rowonly_bigger",             "rowonly_bigger"),
    ("Reverse raster", "generated_rowonly_bigger_revraster",   "rowonly_bigger_revraster"),
    ("Random",         "generated_rowonly_bigger_random",      "rowonly_bigger_random"),
]

# ── Test set mean ─────────────────────────────────────────────────────────────
with h5py.File(os.path.join(data_dir, "fluvial_binary_64x64.h5"), "r") as f:
    imgs_true = f["images"][-1000:]
mean_true = np.mean(imgs_true, axis=0)

# ── Find files ────────────────────────────────────────────────────────────────
def epoch_files(folder, tag):
    """Return {epoch: path} for files named <tag>_epoch_XXXX_n*.npz."""
    pat = re.compile(rf"^{re.escape(tag)}_epoch_(\d+)_n\d+\.npz$")
    found = {}
    folder = os.path.join(data_dir, folder)
    if os.path.isdir(folder):
        for fn in os.listdir(folder):
            m = pat.match(fn)
            if m:
                found[int(m.group(1))] = os.path.join(folder, fn)
    return found

files = {title: epoch_files(folder, tag) for title, folder, tag in ORDERS}
for title, found in files.items():
    print(f"{title}: epochs {sorted(found)}")

common = set.intersection(*(set(found) for found in files.values()))
if EPOCH is None:
    if not common:
        raise FileNotFoundError("No epoch exists for all three orders (see list above)")
    epoch = max(common)
else:
    epoch = EPOCH
    for title, found in files.items():
        if epoch not in found:
            raise FileNotFoundError(f"{title}: no file for epoch {epoch}")
print(f"Plotting epoch {epoch}")

# ── Differences ───────────────────────────────────────────────────────────────
diffs = []
for title, _, _ in ORDERS:
    imgs = np.load(files[title][epoch])["images"]
    mean = np.mean(imgs, axis=0)
    if mean.shape != mean_true.shape:
        raise ValueError(f"{title}: mean shape {mean.shape} != test mean shape {mean_true.shape}")
    diffs.append(mean - mean_true)
    print(f"{title}: mean |diff| = {np.abs(diffs[-1]).mean():.4f}")

vmax = max(np.abs(d).max() for d in diffs)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
im = None
for ax, (title, _, _), diff in zip(axes, ORDERS, diffs):
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_title(f"{title}\nmean |diff| = {np.abs(diff).mean():.4f}")
    ax.axis("off")
fig.colorbar(im, ax=axes, shrink=0.8, label="ViT prior mean − test set mean")

savename = os.path.join(out_dir, f"diff_by_order_epoch_{epoch:04d}.png")
fig.savefig(savename, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"Saved {savename}")

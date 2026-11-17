#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures_gaussexp", exist_ok=True)

base = "data/gauss_exp"
tag  = "A_wells"

data = np.load(f"{base}/obs_{tag}.npz")
npz  = np.load(f"{base}/vit_{tag}.npz")

obs_rows = data["obs_rows"]   # (N, K)
obs_cols = data["obs_cols"]   # (N, K)
obs_vals = data["obs_vals"]   # (N, K)

imgs           = npz["samples"]          # (n, N, H, W)
log_lik_cond   = npz["log_lik_cond"]
log_lik_uncond = npz["log_lik_uncond"]
log_marginal_sum = npz["log_marginal_sum"]

n, N, H, W = imgs.shape

# IS weights
log_w = log_lik_uncond - log_lik_cond - log_marginal_sum
log_w -= log_w.max(axis=1, keepdims=True)
w      = np.exp(log_w)
w     /= w.sum(axis=1, keepdims=True)

mean_weighted = (w[:, :, None, None] * imgs).sum(axis=1)

# i_show = 2 (matches plot_gauss_exp.py for A_wells)
i_show = 2
img = mean_weighted[i_show]

# Two unique well column indices
well_cols = np.unique(obs_cols[i_show])

fig, ax = plt.subplots(figsize=(2, 3))
ax.imshow(img, cmap="gray", interpolation="none", origin="upper", vmin=0, vmax=1)
for col in well_cols:
    ax.axvline(col, color="cyan", linewidth=0.8, alpha=0.8)
ax.axis("off")

plt.savefig("figures_gaussexp/a_wells_isvit.png", dpi=300, bbox_inches="tight", pad_inches=0.15)
plt.savefig("figures_gaussexp/a_wells_isvit.pdf", bbox_inches="tight", pad_inches=0.15)
plt.close()

print("Saved figures_gaussexp/a_wells_isvit.{png,pdf}")
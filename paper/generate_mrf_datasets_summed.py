# -*- coding: utf-8 -*-
"""
generate_mrf_datasets_summed.py

Summed-observation datasets for all four MRF models, single-well regime.

The observation for each image is the EXACT sum of the binary labels in one
column (the "well"): d = sum_r x_{r, WELL_COL}, an integer in [0, n1]. There is
no pixel-level observation and no Gaussian noise -- only the column total.

This matches the masked exact baseline in masked_summed_MRF.py
(exact_marginals_masked* / generate_MRFs_masked*), which condition on
(obs_col, obs_sum), and supplies what the ViT summed proposal needs
(the well pixel indices + the target sum).

Output .npz keys (one file per model, at data/summed_exp/obs_{M}_well.npz):
  images       : (N, n1, n2) uint8  -- sampled MRF images (ground truth)
  obs_col      : int                -- 0-based well column index
  obs_sum      : (N,) int32         -- observed column sum per image
  well_rows    : (n1,) int32        -- row indices of the well pixels (0..n1-1)
  well_cols    : (n1,) int32        -- col indices of the well pixels (= obs_col)
  vit_indices  : (n1,) int64        -- flat row-major indices of the well pixels
  bet, gamma_ne, gamma_nw : model parameters (for self-documentation)

The well must be a column along the n1 (buffer) axis with 1 <= obs_col <= n2-1,
matching masked_summed_MRF (column 0 is unsupported; no axis swap is applied).
"""
import os as _os_shim, sys as _sys_shim
_sys_shim.path.insert(0, _os_shim.path.join(_os_shim.path.dirname(_os_shim.path.abspath(__file__)), "..", "common"))

import numpy as np

from MRF_functions import generate_samples as gen1
from MRF_functions_2ndorder_aniso import generate_samples as gen2

# ── Config ────────────────────────────────────────────────────────────────────
N        = 500
N1, N2   = 12, 24
WELL_COL = 12                 # 0-based well column (interior; set to 6 or 18 to
                              # match the two-well experiments if preferred)
OUT_DIR  = "data/summed_exp"


def _alf_zero():
    return np.zeros((N1, N2, 2), dtype=float)


def _alf_gradient():
    a = np.zeros((N1, N2, 2), dtype=float)
    for r in range(N1):
        a[r, :, 0] = np.linspace(-0.25, 0.25, N2)
    return a


# (name, order, alf, bet, gamma_ne, gamma_nw, seed)  -- fresh seeds
MODELS = [
    dict(name="A", order=1, alf=_alf_zero(),     bet=0.7, gne=0.0, gnw=0.0, seed=1280),
    dict(name="B", order=1, alf=_alf_gradient(), bet=0.7, gne=0.0, gnw=0.0, seed=1050),
    dict(name="C", order=2, alf=_alf_zero(),     bet=0.5, gne=0.2, gnw=0.2, seed=1170),
    dict(name="D", order=2, alf=_alf_gradient(), bet=0.5, gne=0.3, gnw=0.1, seed=1490),
]

if not (1 <= WELL_COL <= N2 - 1):
    raise ValueError(f"WELL_COL must be in [1, {N2 - 1}] (0-based, interior/last column).")

well_rows   = np.arange(N1, dtype=np.int32)
well_cols   = np.full(N1, WELL_COL, dtype=np.int32)
vit_indices = (well_rows.astype(np.int64) * N2 + WELL_COL)

for m in MODELS:
    if m["order"] == 1:
        images = gen1(m["alf"], m["bet"], N, seed=m["seed"], dtype=np.uint8)
    else:
        images = gen2(m["alf"], m["bet"], N,
                      gamma_ne=m["gne"], gamma_nw=m["gnw"],
                      seed=m["seed"], dtype=np.uint8)

    # exact summed observation: number of 1s in the well column, per image
    obs_sum = images[:, :, WELL_COL].sum(axis=1).astype(np.int32)   # (N,)

    out_path = f"{OUT_DIR}/obs_{m['name']}_well.npz"
    np.savez_compressed(
        out_path,
        images      = images,
        obs_col     = np.int64(WELL_COL),
        obs_sum     = obs_sum,
        well_rows   = well_rows,
        well_cols   = well_cols,
        vit_indices = vit_indices,
        bet         = float(m["bet"]),
        gamma_ne    = float(m["gne"]),
        gamma_nw    = float(m["gnw"]),
    )
    print(f"Model {m['name']}: saved {N} images to {out_path}  "
          f"(well col {WELL_COL}, obs_sum range [{obs_sum.min()}, {obs_sum.max()}], "
          f"mean {obs_sum.mean():.2f})")

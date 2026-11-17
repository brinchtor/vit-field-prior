"""
generate_mrf_conds_A_random.py

Load a dataset produced by generate_mrf_datasets_A.py, draw n_samples exact
posterior samples for each image (conditioning by hard-clamping observed pixels
via -inf unary log-potentials), and save the results.

Output: data/cond_exp/mrf_A_random.npz
    images   float64  [N, n_samples, n1, n2]
"""
import os as _os_shim, sys as _sys_shim
_sys_shim.path.insert(0, _os_shim.path.join(_os_shim.path.dirname(_os_shim.path.abspath(__file__)), "..", "common"))

import numpy as np
from MRF_functions import generate_samples

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = "data/cond_exp/obs_A_random.npz"
BET       = 0.7
OUT_PATH  = "data/cond_exp/mrf_A_random.npz"

# ── Load ──────────────────────────────────────────────────────────────────────
data = np.load(DATA_PATH)
alf  = np.zeros((12, 24, 2), dtype=float)
bet = 0.7

obs_rows = data["obs_rows"]   # (N, K)
obs_cols = data["obs_cols"]   # (N, K)
obs_vals = data["obs_vals"]   # (N, K)
N        = obs_rows.shape[0]
K = obs_rows.shape[1]
n1, n2   = alf.shape[:2]
n_samples = 1000

images = np.zeros((N, n_samples, n1, n2))

for i in range(N):
    print(f"Image {i+1}/{N} ...", flush=True)
    alf  = np.zeros((12, 24, 2), dtype=float)
    for j in range(K):
        alf[obs_rows[i,j], obs_cols[i,j], 1 - obs_vals[i,j]] = -np.inf
    images[i] = generate_samples(alf, bet, n_samples, seed=i)

np.savez_compressed(OUT_PATH, images=images)
print(f"Saved to {OUT_PATH}")
"""
generate_mrf_conds_gauss_B_random.py

Gaussian-noise exact MRF conditional samples for model B (first-order,
nonstationary; column-varying unary), random regime. Loads the noisy
observations, folds the Gaussian log-likelihood -(y_i - l)^2 / (2 sigma^2)
into the unary at each observed pixel (on top of the prior gradient), draws
exact conditional samples, and saves them.

Output: data/gauss_exp/mrf_B_random.npz
    images   float64  [N, n_samples, n1, n2]
"""
import os as _os_shim, sys as _sys_shim
_sys_shim.path.insert(0, _os_shim.path.join(_os_shim.path.dirname(_os_shim.path.abspath(__file__)), "..", "common"))

import numpy as np
from MRF_functions import generate_samples

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = "data/gauss_exp/obs_B_random.npz"
BET       = 0.7
SIGMA     = 0.5
OUT_PATH  = "data/gauss_exp/mrf_B_random.npz"

# ── Load ──────────────────────────────────────────────────────────────────────
data = np.load(DATA_PATH)
bet   = BET
sigma = SIGMA

obs_rows = data["obs_rows"]   # (N, K)
obs_cols = data["obs_cols"]   # (N, K)
obs_vals = data["obs_vals"]   # (N, K)  noisy y
N        = obs_rows.shape[0]
K        = obs_rows.shape[1]
n1, n2   = 12, 24
n_samples = 1000

images = np.zeros((N, n_samples, n1, n2))

for i in range(N):
    print(f"Image {i+1}/{N} ...", flush=True)
    alf = np.zeros((n1, n2, 2), dtype=float)
    for rr in range(n1):
        alf[rr, :, 0] = np.linspace(-0.25, 0.25, n2)
    for j in range(K):
        alf[obs_rows[i, j], obs_cols[i, j], 0] += -(obs_vals[i, j] - 0)**2 / (2 * sigma**2)
        alf[obs_rows[i, j], obs_cols[i, j], 1] += -(obs_vals[i, j] - 1)**2 / (2 * sigma**2)
    images[i] = generate_samples(alf, bet, n_samples, seed=i)

np.savez_compressed(OUT_PATH, images=images)
print(f"Saved to {OUT_PATH}")

"""
generate_mrf_conds_gauss_C_wells.py

Gaussian-noise exact MRF conditional samples for model C (second-order
anisotropic, stationary), wells regime. Uses the second-order sampler with
diagonal couplings gamma_ne / gamma_nw. Folds the Gaussian log-likelihood
-(y_i - l)^2 / (2 sigma^2) into the unary at each observed pixel, draws exact
conditional samples, and saves them.

Output: data/gauss_exp/mrf_C_wells.npz
    images   float64  [N, n_samples, n1, n2]
"""
import os as _os_shim, sys as _sys_shim
_sys_shim.path.insert(0, _os_shim.path.join(_os_shim.path.dirname(_os_shim.path.abspath(__file__)), "..", "common"))

import numpy as np
from MRF_functions_2ndorder_aniso import generate_samples

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = "data/gauss_exp/obs_C_wells.npz"
BET       = 0.5
GAMMA_NE  = 0.2
GAMMA_NW  = 0.2
SIGMA     = 0.5
OUT_PATH  = "data/gauss_exp/mrf_C_wells.npz"

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
    for j in range(K):
        alf[obs_rows[i, j], obs_cols[i, j], 0] += -(obs_vals[i, j] - 0)**2 / (2 * sigma**2)
        alf[obs_rows[i, j], obs_cols[i, j], 1] += -(obs_vals[i, j] - 1)**2 / (2 * sigma**2)
    images[i] = generate_samples(
        alf, bet, n_samples, gamma_ne=GAMMA_NE, gamma_nw=GAMMA_NW, seed=i
    )

np.savez_compressed(OUT_PATH, images=images)
print(f"Saved to {OUT_PATH}")

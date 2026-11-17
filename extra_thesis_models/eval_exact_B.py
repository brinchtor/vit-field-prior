"""
eval_exact_B.py

Load a dataset produced by generate_mrf_datasets_B.py, compute exact posterior
marginals for each image, and save the results.

Output: data/cond_exp/exact_B_wells.npz
    marginals   float64  [N, n1, n2, 2]  -- P(x_{r,c}=v | y) for each image
    logZ        float64  [N]             -- log p(y) for each image
"""
import os as _os_shim, sys as _sys_shim
_sys_shim.path.insert(0, _os_shim.path.join(_os_shim.path.dirname(_os_shim.path.abspath(__file__)), "..", "common"))

import numpy as np
import exact_marginals

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = "data/cond_exp/obs_B_wells.npz"
BET       = 0.7
OUT_PATH  = "data/cond_exp/exact_B_wells.npz"

# ── Load ──────────────────────────────────────────────────────────────────────
data = np.load(DATA_PATH)
alf  = np.zeros((12, 24, 2), dtype=float)
for i in range(12):
    alf[i,:,0] = np.linspace(-0.25,0.25,24)

obs_rows = data["obs_rows"]   # (N, K)
obs_cols = data["obs_cols"]   # (N, K)
obs_vals = data["obs_vals"]   # (N, K)
N        = obs_rows.shape[0]
n1, n2   = alf.shape[:2]

# ── Compute marginals ─────────────────────────────────────────────────────────
all_marginals = np.empty((N, n1, n2, 2), dtype=np.float64)
all_logZ      = np.empty(N,              dtype=np.float64)

for i in range(N):
    print(f"Image {i+1}/{N} ...", flush=True)
    obs_dict = {
        (int(r), int(c)): int(v)
        for r, c, v in zip(obs_rows[i], obs_cols[i], obs_vals[i])
    }
    marg, logZ = exact_marginals.marginals(
        alf, BET, observations=obs_dict, return_logZ=True
    )
    all_marginals[i] = marg
    all_logZ[i]      = logZ

# ── Save ──────────────────────────────────────────────────────────────────────
np.savez_compressed(OUT_PATH, marginals=all_marginals, logZ=all_logZ)
print(f"Saved to {OUT_PATH}")
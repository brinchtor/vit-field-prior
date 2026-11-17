"""
eval_exact_gauss_all.py

Gaussian-noise exact posterior marginals for all four MRF benchmark models
(A-D) and both observation regimes (random, wells). For each image the
Gaussian log-likelihood -(y_i - l)^2 / (2 sigma^2) is folded into the unary
alf at each observed pixel (on top of the model's prior unary), then exact
marginals are computed with observations=None.

Output (one file per config): data/gauss_exp/exact_{model}_{regime}.npz
    marginals   float64  [N, n1, n2, 2]  -- P(x_{r,c}=v | y) for each image
    logZ        float64  [N]             -- log p(y) for each image
"""
import os as _os_shim, sys as _sys_shim
_sys_shim.path.insert(0, _os_shim.path.join(_os_shim.path.dirname(_os_shim.path.abspath(__file__)), "..", "common"))

import numpy as np
import exact_marginals
import exact_marginals_2ndorder_aniso

# ── Config ────────────────────────────────────────────────────────────────────
SIGMA  = 0.5
N1, N2 = 12, 24

# (model, regime, order, bet, gamma_ne, gamma_nw, nonstationary)
CONFIGS = [
    ("A", "random", 1, 0.7, None, None, False),
    ("A", "wells",  1, 0.7, None, None, False),
    ("B", "random", 1, 0.7, None, None, True),
    ("B", "wells",  1, 0.7, None, None, True),
    ("C", "random", 2, 0.5, 0.2,  0.2,  False),
    ("C", "wells",  2, 0.5, 0.2,  0.2,  False),
    ("D", "random", 2, 0.5, 0.3,  0.1,  True),
    ("D", "wells",  2, 0.5, 0.3,  0.1,  True),
]


def base_unary(nonstationary):
    """Prior unary: zeros, plus a per-row column gradient for the nonstationary models."""
    alf = np.zeros((N1, N2, 2), dtype=float)
    if nonstationary:
        for rr in range(N1):
            alf[rr, :, 0] = np.linspace(-0.25, 0.25, N2)
    return alf


# ── Run ───────────────────────────────────────────────────────────────────────
for model, regime, order, bet, g_ne, g_nw, nonstat in CONFIGS:
    data_path = f"data/gauss_exp/obs_{model}_{regime}.npz"
    out_path  = f"data/gauss_exp/exact_{model}_{regime}.npz"
    print(f"=== {model}_{regime} ===", flush=True)

    data     = np.load(data_path)
    obs_rows = data["obs_rows"]   # (N, K)
    obs_cols = data["obs_cols"]   # (N, K)
    obs_vals = data["obs_vals"]   # (N, K)  noisy y
    N, K     = obs_rows.shape

    all_marginals = np.empty((N, N1, N2, 2), dtype=np.float64)
    all_logZ      = np.empty(N,              dtype=np.float64)

    for i in range(N):
        print(f"Image {i+1}/{N} ...", flush=True)
        alf = base_unary(nonstat)
        for j in range(K):
            r, c, y = obs_rows[i, j], obs_cols[i, j], obs_vals[i, j]
            alf[r, c, 0] += -(y - 0)**2 / (2 * SIGMA**2)
            alf[r, c, 1] += -(y - 1)**2 / (2 * SIGMA**2)

        if order == 1:
            marg, logZ = exact_marginals.marginals(
                alf, bet, observations=None, return_logZ=True
            )
        else:
            marg, logZ = exact_marginals_2ndorder_aniso.marginals(
                alf, bet, gamma_ne=g_ne, gamma_nw=g_nw,
                observations=None, return_logZ=True
            )

        all_marginals[i] = marg
        all_logZ[i]      = logZ

    np.savez_compressed(out_path, marginals=all_marginals, logZ=all_logZ)
    print(f"Saved to {out_path}", flush=True)
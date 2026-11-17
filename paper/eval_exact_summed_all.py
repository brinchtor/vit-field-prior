"""
eval_exact_summed_all.py

Load the summed-observation datasets (one well, exact column sum) produced by
generate_mrf_datasets_summed.py, compute the EXACT posterior marginals for each
image given sum(column obs_col) = obs_sum[i], and save them -- the analytical
ground truth E[x | d] used for the MSE evaluation.

Uses the masked single-column-sum marginals in masked_summed_MRF.py:
  * models A, B  -> exact_marginals_masked                 (first order)
  * models C, D  -> exact_marginals_masked_2ndorder_aniso  (second order)

Output (one file per model): data/summed_exp/exact_{M}_well.npz
    marginals   float64  [N, n1, n2, 2]  -- P(x_{r,c}=v | d) for each image
    logZ        float64  [N]             -- log conditioned partition mass
"""
import os as _os_shim, sys as _sys_shim
_sys_shim.path.insert(0, _os_shim.path.join(_os_shim.path.dirname(_os_shim.path.abspath(__file__)), "..", "common"))

import numpy as np

from masked_summed_MRF import (
    exact_marginals_masked,
    exact_marginals_masked_2ndorder_aniso,
)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = "data/summed_exp"
N1, N2   = 12, 24


def _alf_zero():
    return np.zeros((N1, N2, 2), dtype=float)


def _alf_gradient():
    a = np.zeros((N1, N2, 2), dtype=float)
    for r in range(N1):
        a[r, :, 0] = np.linspace(-0.25, 0.25, N2)
    return a


# (name, order, alf, bet, gamma_ne, gamma_nw) -- must match the dataset generator
MODELS = [
    dict(name="A", order=1, alf=_alf_zero(),     bet=0.7, gne=0.0, gnw=0.0),
    dict(name="B", order=1, alf=_alf_gradient(), bet=0.7, gne=0.0, gnw=0.0),
    dict(name="C", order=2, alf=_alf_zero(),     bet=0.5, gne=0.2, gnw=0.2),
    dict(name="D", order=2, alf=_alf_gradient(), bet=0.5, gne=0.3, gnw=0.1),
]

for m in MODELS:
    data    = np.load(f"{DATA_DIR}/obs_{m['name']}_well.npz")
    obs_col = int(data["obs_col"])
    obs_sum = data["obs_sum"]          # (N,) exact well-column sums
    N       = obs_sum.shape[0]

    all_marginals = np.empty((N, N1, N2, 2), dtype=np.float64)
    all_logZ      = np.empty(N,             dtype=np.float64)

    for i in range(N):
        print(f"Model {m['name']} — image {i+1}/{N} ...", flush=True)
        if m["order"] == 1:
            marg, logZ = exact_marginals_masked(
                m["alf"], m["bet"], obs_col, int(obs_sum[i]), return_logZ=True
            )
        else:
            marg, logZ = exact_marginals_masked_2ndorder_aniso(
                m["alf"], m["bet"], obs_col, int(obs_sum[i]),
                gamma_ne=m["gne"], gamma_nw=m["gnw"], return_logZ=True
            )
        all_marginals[i] = marg
        all_logZ[i]      = logZ

    out_path = f"{DATA_DIR}/exact_{m['name']}_well.npz"
    np.savez_compressed(out_path, marginals=all_marginals, logZ=all_logZ)
    print(f"Saved to {out_path}")

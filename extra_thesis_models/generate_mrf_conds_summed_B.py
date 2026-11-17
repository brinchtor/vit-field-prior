"""
generate_mrf_conds_summed_B.py

Load the summed-observation dataset (one well, exact column sum) produced by
generate_mrf_datasets_summed.py, draw exact conditional MRF samples for each
image given sum(column obs_col) = obs_sum[i], and save them. Uses the masked
single-column-sum sampler in masked_summed_MRF.py.

Output: data/summed_exp/mrf_B_well.npz
    images   float64  [N, n_samples, n1, n2]
"""
import os as _os_shim, sys as _sys_shim
_sys_shim.path.insert(0, _os_shim.path.join(_os_shim.path.dirname(_os_shim.path.abspath(__file__)), "..", "common"))

import numpy as np
from masked_summed_MRF import generate_MRFs_masked

# -- Config --------------------------------------------------------------------
DATA_PATH = "data/summed_exp/obs_B_well.npz"
BET       = 0.7
OUT_PATH  = "data/summed_exp/mrf_B_well.npz"

# -- Load ----------------------------------------------------------------------
data    = np.load(DATA_PATH)
obs_col = int(data["obs_col"])
obs_sum = data["obs_sum"]          # (N,) exact well-column sums
N       = obs_sum.shape[0]
n1, n2  = 12, 24
n_samples = 1000

# Prior unary (the sum conditioning is applied inside the sampler, not via alf)
alf = np.zeros((n1, n2, 2), dtype=float)
for r in range(n1):
    alf[r, :, 0] = np.linspace(-0.25, 0.25, n2)

images = np.zeros((N, n_samples, n1, n2))

for i in range(N):
    print(f"Image {i+1}/{N} ...", flush=True)
    images[i] = generate_MRFs_masked(
        alf, BET, n_samples, obs_col, int(obs_sum[i]), seed=i
    )

np.savez_compressed(OUT_PATH, images=images)
print(f"Saved to {OUT_PATH}")

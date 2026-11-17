# -*- coding: utf-8 -*-
"""
Driver for the second-order MRF sampler.

Set gamma to 0.0 to recover the first-order Ising/Potts model;
set gamma != 0.0 to add the NE and SE diagonal couplings (2x2 cliques).
"""
import os as _os_shim, sys as _sys_shim
_sys_shim.path.insert(0, _os_shim.path.join(_os_shim.path.dirname(_os_shim.path.abspath(__file__)), "..", "common"))

import os
import numpy as np
from MRF_functions_2ndorder import generate_samples_npz

# ---- Grid and model parameters ----
n1, n2, d = 12, 24, 2
alf = np.zeros((n1, n2, d), dtype=float)
bet = 0.5
gamma = 0.2   # <-- edit this

# ---- Sampling parameters ----
N = 5*10**2
seed = 42

# ---- Output ----
out_dir = "data"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"mrf-model-C.npz")

generate_samples_npz(
    alf=alf,
    bet=bet,
    N=N,
    out_path=out_path,
    gamma=gamma,
    seed=seed,
    force_n1_smaller=True,
    dtype=np.uint16,
)

print(f"Saved {N} samples to {out_path} (bet={bet}, gamma={gamma})")
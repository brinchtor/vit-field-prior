"""
get_marginals.py

Compute the exact UNCONDITIONAL (prior) marginals of models A, B, C and D
(no observations) and save them.

Output: data/marginals.npz
    A, B, C, D   float64  [n1, n2, 2]  -- P(x_{r,c}=v) under each model
"""
import os as _os_shim, sys as _sys_shim
_sys_shim.path.insert(0, _os_shim.path.join(_os_shim.path.dirname(_os_shim.path.abspath(__file__)), "..", "common"))

import numpy as np
import exact_marginals
import exact_marginals_2ndorder_aniso

OUT_PATH  = "data/marginals.npz"

# ── Model A ────────────────────────────────────────────────────────────────────
alf  = np.zeros((12, 24, 2), dtype=float)
BET       = 0.7

marginals_A = exact_marginals.marginals(
    alf, BET, observations=None
)

# ── Model B ────────────────────────────────────────────────────────────────────
alf_B  = np.zeros((12, 24, 2), dtype=float)
for r in range(12):
    alf_B[r, :, 0] = np.linspace(-0.25, 0.25, 24)

marginals_B = exact_marginals.marginals(
    alf_B, BET, observations=None
)

# ── Model C ────────────────────────────────────────────────────────────────────
BET = 0.5
GAMMA_NE = 0.2
GAMMA_NW = 0.2

marginals_C = exact_marginals_2ndorder_aniso.marginals(
    alf, BET, gamma_ne=GAMMA_NE, gamma_nw=GAMMA_NW,
    observations=None
)

# ── Model D ────────────────────────────────────────────────────────────────────
GAMMA_NE = 0.3
GAMMA_NW = 0.1

marginals_D = exact_marginals_2ndorder_aniso.marginals(
    alf_B, BET, gamma_ne=GAMMA_NE, gamma_nw=GAMMA_NW,
    observations=None
)

# ── Save ──────────────────────────────────────────────────────────────────────
np.savez_compressed(OUT_PATH, A=marginals_A, B=marginals_B, C=marginals_C, D=marginals_D)
print(f"Saved to {OUT_PATH}")
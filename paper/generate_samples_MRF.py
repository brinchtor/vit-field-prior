# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 14:58:02 2026

@author: brinc
"""
import os as _os_shim, sys as _sys_shim
_sys_shim.path.insert(0, _os_shim.path.join(_os_shim.path.dirname(_os_shim.path.abspath(__file__)), "..", "common"))

import numpy as np
from MRF_functions import generate_samples_npz

n1, n2, d = 12, 24, 2
alf = np.zeros((n1, n2, d), dtype=float)
bet = 0.7

N = 10**5
out_path = "data/mrf-model-A-train.npz"

generate_samples_npz(
    alf=alf,
    bet=bet,
    N=N,
    out_path=out_path,
    seed=0,
    force_n1_smaller=True,
    dtype=np.uint16,
)

print(f"Saved {N} samples to {out_path}")

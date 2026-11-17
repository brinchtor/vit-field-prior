# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 14:58:02 2026

@author: brinc
"""
import os as _os_shim, sys as _sys_shim
_sys_shim.path.insert(0, _os_shim.path.join(_os_shim.path.dirname(_os_shim.path.abspath(__file__)), "..", "common"))

import numpy as np
from MRF_functions_2ndorder_aniso import generate_samples_npz
import matplotlib.pyplot as plt

n1, n2, d = 12, 24, 2
alf = np.zeros((n1, n2, d), dtype=float)
for i in range(n1):
    alf[i,:,0] = np.linspace(-0.25,0.25,n2)
bet = 0.5
gamma_ne = 0.3
gamma_nw = 0.1

N = 10**3
out_path = "data/mrf-model-D.npz"

generate_samples_npz(
    alf=alf,
    bet=bet,
    N=N,
    out_path=out_path,
    seed=42,
    gamma_ne=gamma_ne,
    gamma_nw=gamma_nw,
    force_n1_smaller=True,
    dtype=np.uint16,
)

print(f"Saved {N} samples to {out_path}")

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 14:58:02 2026

@author: brinc
"""

import numpy as np
from MRF_functions_nonstat import generate_samples_npz
import matplotlib.pyplot as plt

n1, n2, d = 12, 24, 2
alf = np.zeros((n1, n2, d), dtype=float)
for i in range(n1):
    alf[i,:,0] = np.linspace(-0.25,0.25,n2)
bet = 0.7

N = 5*10**2
out_path = "data/mrf-model-B.npz"

generate_samples_npz(
    alf=alf,
    bet=bet,
    N=N,
    out_path=out_path,
    seed=42,
    force_n1_smaller=True,
    dtype=np.uint16,
)

print(f"Saved {N} samples to {out_path}")

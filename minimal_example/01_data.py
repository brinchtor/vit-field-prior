"""
01_data.py

1. Draw exact model-A samples with common/MRF_functions.generate_samples (the
   sampler behind paper/generate_samples_MRF.py) and split them into a small
   training set (label-flip augmented and shuffled, as in
   MRF_functions.generate_samples_npz) and a held-out test set.
2. Create small posterior test sets with paper/generate_mrf_datasets_A.py:
   "random" (N_RANDOM_OBS random pixels) and "wells" (columns WELL_COLS),
   exact (noise-free) observations.

Output: out/mrf_A_train.npz, out/mrf_A_test.npz,
        out/obs_A_random.npz, out/obs_A_wells.npz
"""
import time

import numpy as np

from run_all import (BET, N1, N_OBS_IMAGES, N_RANDOM_OBS, N_TEST, N_TRAIN, SEED,
                     WELL_COLS, alf, out)
from MRF_functions import generate_samples
from generate_mrf_datasets_A import generate_fixed_observations, generate_random_observations

t0 = time.time()
samples, _, logZ, _, _ = generate_samples(alf(), BET, N_TRAIN + N_TEST, seed=SEED,
                                          return_all=True, dtype=np.uint8)
train, test = samples[:N_TRAIN], samples[N_TRAIN:]
train = np.concatenate([train, 1 - train], axis=0)
train = train[np.random.default_rng(SEED).permutation(len(train))]
np.savez_compressed(out("mrf_A_train.npz"), images=train, logZ=logZ, bet=BET)
np.savez_compressed(out("mrf_A_test.npz"), images=test, logZ=logZ, bet=BET)
print(f"exact logZ = {logZ:.4f}; {len(train)} training / {len(test)} test images")

generate_random_observations(alf(), BET, n_observed=N_RANDOM_OBS, n=N_OBS_IMAGES,
                             out_path=out("obs_A_random.npz"), seed=SEED + 50)
generate_fixed_observations(alf(), BET, [(r, c) for c in WELL_COLS for r in range(N1)],
                            n=N_OBS_IMAGES, out_path=out("obs_A_wells.npz"), seed=SEED + 280)
print(f"Done ({time.time() - t0:.1f} s)")

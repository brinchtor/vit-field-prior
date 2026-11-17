# -*- coding: utf-8 -*-
"""
generate_mrf_datasets_gauss_D.py

Gaussian-noise version of generate_mrf_datasets_D.py. Uses the second-order
anisotropic sampler in MRF_functions_2ndorder_aniso.py; the MRF is parameterized
by (alf, bet, gamma_ne, gamma_nw), where gamma_ne / gamma_nw are the NE/SW and
NW/SE diagonal couplings (both 0.0 recovers the first-order Ising/Potts model).
Model D is the nonstationary second-order case (column-varying unary, unequal
diagonal couplings). Each observed pixel is recorded as a noisy real value
y_i = x_i + eps_i, eps_i ~ N(0, sigma^2), rather than the exact 0/1 label.

Output .npz keys (per function):
  images         : (N, n1, n2)  uint8    -- sampled MRF images
  obs_rows       : (N, K)       int32    -- row indices of observed pixels
  obs_cols       : (N, K)       int32    -- col indices of observed pixels
  obs_vals       : (N, K)       float64  -- noisy observed values y_i
  vit_indices    : (N, K)       int64    -- flat row-major indices for ViT
  vit_values     : (N, K)       float64  -- noisy observed values y_i for ViT

For function 1 (fixed pixel set) K = len(observed_pixels) for all images.
For function 2 (random pixel set) K = n_observed for all images.
"""
import os as _os_shim, sys as _sys_shim
_sys_shim.path.insert(0, _os_shim.path.join(_os_shim.path.dirname(_os_shim.path.abspath(__file__)), "..", "common"))

import numpy as np
from MRF_functions_2ndorder_aniso import generate_samples


def generate_fixed_observations(
    alf: np.ndarray,
    bet: float,
    observed_pixels: list[tuple[int, int]],
    n: int,
    sigma: float,
    gamma_ne: float = 0.0,
    gamma_nw: float = 0.0,
    out_path: str | None = None,
    seed: int = 0,
) -> dict:
    """
    Sample n images from the second-order anisotropic MRF and observe a fixed
    set of pixels in each, through a Gaussian noise channel y_i = x_i + N(0, sigma^2).
    """
    n1, n2 = alf.shape[:2]
    K = len(observed_pixels)

    for r, c in observed_pixels:
        if not (0 <= r < n1 and 0 <= c < n2):
            raise ValueError(f"Pixel ({r},{c}) is out of bounds for grid ({n1},{n2}).")

    images = generate_samples(
        alf, bet, n, gamma_ne=gamma_ne, gamma_nw=gamma_nw, seed=seed, dtype=np.uint8
    )  # (N, n1, n2)

    all_obs_rows  = np.empty((n, K), dtype=np.int32)
    all_obs_cols  = np.empty((n, K), dtype=np.int32)
    all_obs_vals  = np.empty((n, K), dtype=np.float64)
    all_vit_idx   = np.empty((n, K), dtype=np.int64)
    all_vit_vals  = np.empty((n, K), dtype=np.float64)

    rows_arr = np.array([r for r, c in observed_pixels], dtype=np.int32)
    cols_arr = np.array([c for r, c in observed_pixels], dtype=np.int32)
    idx_arr  = np.array([r * n2 + c for r, c in observed_pixels], dtype=np.int64)

    rng_norm = np.random.default_rng(seed + 2)

    for i, img in enumerate(images):
        vals = img[rows_arr, cols_arr].astype(np.float64) + rng_norm.normal(loc=0, scale=sigma, size=K)
        all_obs_rows[i] = rows_arr
        all_obs_cols[i] = cols_arr
        all_obs_vals[i] = vals
        all_vit_idx[i]  = idx_arr
        all_vit_vals[i] = vals

    result = dict(
        images       = images,
        obs_rows     = all_obs_rows,
        obs_cols     = all_obs_cols,
        obs_vals     = all_obs_vals,
        vit_indices  = all_vit_idx,
        vit_values   = all_vit_vals,
        observed_pixels = observed_pixels,
    )

    if out_path is not None:
        np.savez_compressed(
            out_path,
            images      = images,
            obs_rows    = all_obs_rows,
            obs_cols    = all_obs_cols,
            obs_vals    = all_obs_vals,
            vit_indices = all_vit_idx,
            vit_values  = all_vit_vals,
            obs_rows_fixed = rows_arr,
            obs_cols_fixed = cols_arr,
        )
        print(f"Saved {n} images with {K} fixed observations each to {out_path}")

    return result


def generate_random_observations(
    alf: np.ndarray,
    bet: float,
    n_observed: int,
    n: int,
    sigma: float,
    gamma_ne: float = 0.0,
    gamma_nw: float = 0.0,
    out_path: str | None = None,
    seed: int = 0,
) -> dict:
    """
    Sample n images from the second-order anisotropic MRF and observe a different
    random set of n_observed pixels in each image, through a Gaussian noise channel.
    """
    n1, n2 = alf.shape[:2]
    n_pixels = n1 * n2

    if not (0 < n_observed <= n_pixels):
        raise ValueError(
            f"n_observed={n_observed} must be in [1, {n_pixels}] for a ({n1},{n2}) grid."
        )

    images = generate_samples(
        alf, bet, n, gamma_ne=gamma_ne, gamma_nw=gamma_nw, seed=seed, dtype=np.uint8
    )  # (N, n1, n2)

    all_obs_rows  = np.empty((n, n_observed), dtype=np.int32)
    all_obs_cols  = np.empty((n, n_observed), dtype=np.int32)
    all_obs_vals  = np.empty((n, n_observed), dtype=np.float64)
    all_vit_idx   = np.empty((n, n_observed), dtype=np.int64)
    all_vit_vals  = np.empty((n, n_observed), dtype=np.float64)

    rng = np.random.default_rng(seed + 1)
    rng_norm = np.random.default_rng(seed + 2)
    all_flat = np.arange(n_pixels, dtype=np.int64)

    for i, img in enumerate(images):
        chosen = rng.choice(all_flat, size=n_observed, replace=False)
        chosen.sort()

        rows = (chosen // n2).astype(np.int32)
        cols = (chosen %  n2).astype(np.int32)
        vals = img[rows, cols].astype(np.float64) + rng_norm.normal(loc=0, scale=sigma, size=n_observed)

        all_obs_rows[i] = rows
        all_obs_cols[i] = cols
        all_obs_vals[i] = vals
        all_vit_idx[i]  = chosen
        all_vit_vals[i] = vals

    result = dict(
        images      = images,
        obs_rows    = all_obs_rows,
        obs_cols    = all_obs_cols,
        obs_vals    = all_obs_vals,
        vit_indices = all_vit_idx,
        vit_values  = all_vit_vals,
        observed_pixels = None,  # random per image; see obs_rows/obs_cols
    )

    if out_path is not None:
        np.savez_compressed(
            out_path,
            images      = images,
            obs_rows    = all_obs_rows,
            obs_cols    = all_obs_cols,
            obs_vals    = all_obs_vals,
            vit_indices = all_vit_idx,
            vit_values  = all_vit_vals,
        )
        print(f"Saved {n} images with {n_observed} random observations each to {out_path}")

    return result


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n1, n2, d = 12, 24, 2
    alf = np.zeros((n1, n2, d), dtype=float)
    for i in range(12):
        alf[i, :, 0] = np.linspace(-0.25, 0.25, 24)
    bet = 0.5
    gamma_ne = 0.3
    gamma_nw = 0.1
    sigma = 0.5

    # Function 1: observe all pixels in columns 6 and 18
    fixed_pixels = [(r, c) for c in [6, 18] for r in range(n1)]
    d1 = generate_fixed_observations(
        alf, bet, fixed_pixels, n=500, sigma=sigma,
        gamma_ne=gamma_ne, gamma_nw=gamma_nw,
        out_path="data/gauss_exp/obs_D_wells.npz", seed=47
    )
    print("Fixed obs — images:", d1["images"].shape,
          "vit_indices:", d1["vit_indices"].shape)

    # Function 2: observe 30 random pixels per image
    d2 = generate_random_observations(
        alf, bet, n_observed=30, n=500, sigma=sigma,
        gamma_ne=gamma_ne, gamma_nw=gamma_nw,
        out_path="data/gauss_exp/obs_D_random.npz", seed=490
    )
    print("Random obs — images:", d2["images"].shape,
          "vit_indices:", d2["vit_indices"].shape)

# -*- coding: utf-8 -*-
"""
generate_mrf_datasets_gauss_A.py

Two functions for generating MRF image datasets with associated observations,
formatted for both exact_marginals.py and the ViT conditional sampler.

Observation formats produced for each image:
  - obs_dict  : {(row, col): value}  -- 0-based (r,c), as exact_marginals.marginals() expects
  - vit_indices : np.ndarray int64   -- flat row-major pixel indices (row*W + col)
  - vit_values  : np.ndarray int64   -- corresponding pixel values

Output .npz keys (per function):
  images         : (N, n1, n2)  uint8  -- sampled MRF images
  obs_rows       : (N, K)       int32  -- row indices of observed pixels
  obs_cols       : (N, K)       int32  -- col indices of observed pixels
  obs_vals       : (N, K)       float64  -- observed pixel values (0/1)
  vit_indices    : (N, K)       int64  -- flat row-major indices for ViT
  vit_values     : (N, K)       float64  -- pixel values for ViT

For function 1 (fixed pixel set) K = len(observed_pixels) for all images.
For function 2 (random pixel set) K = n_observed for all images.
"""
import os as _os_shim, sys as _sys_shim
_sys_shim.path.insert(0, _os_shim.path.join(_os_shim.path.dirname(_os_shim.path.abspath(__file__)), "..", "common"))

import numpy as np
from MRF_functions import generate_samples


def _build_obs_arrays(image: np.ndarray, pixel_coords: list[tuple[int, int]]):
    """
    Given a 2D image and a list of (row, col) coordinates, return:
      obs_dict     : {(row, col): value}  for exact_marginals
      vit_indices  : int64 array of flat row-major indices
      vit_values   : int64 array of pixel values
    """
    n1, n2 = image.shape
    obs_dict = {}
    rows, cols, vals = [], [], []
    for r, c in pixel_coords:
        v = int(image[r, c])
        obs_dict[(r, c)] = v
        rows.append(r)
        cols.append(c)
        vals.append(v)
    vit_indices = np.array([r * n2 + c for r, c in zip(rows, cols)], dtype=np.int64)
    vit_values  = np.array(vals, dtype=np.int64)
    return obs_dict, np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32), vit_indices, vit_values


def generate_fixed_observations(
    alf: np.ndarray,
    bet: float,
    observed_pixels: list[tuple[int, int]],
    n: int,
    sigma: float,
    out_path: str | None = None,
    seed: int = 0,
) -> dict:
    """
    Sample n images from the MRF and observe a fixed set of pixels in each.

    Parameters
    ----------
    alf : (n1, n2, d) unary log-potentials
    bet : scalar coupling
    observed_pixels : list of (row, col) pairs to observe in every image (0-based)
    n : number of images to generate
    out_path : if given, save output to this .npz path
    seed : RNG seed for MRF sampler

    Returns
    -------
    dict with keys:
      images        (N, n1, n2) uint8
      obs_rows      (N, K) int32
      obs_cols      (N, K) int32
      obs_vals      (N, K) float64
      vit_indices   (N, K) int64
      vit_values    (N, K) float64
      observed_pixels  list[(row,col)]  -- the fixed pixel set used
    """
    n1, n2 = alf.shape[:2]
    K = len(observed_pixels)

    # Validate coordinates
    for r, c in observed_pixels:
        if not (0 <= r < n1 and 0 <= c < n2):
            raise ValueError(f"Pixel ({r},{c}) is out of bounds for grid ({n1},{n2}).")

    images = generate_samples(alf, bet, n, seed=seed, dtype=np.uint8)  # (N, n1, n2)

    all_obs_rows  = np.empty((n, K), dtype=np.int32)
    all_obs_cols  = np.empty((n, K), dtype=np.int32)
    all_obs_vals  = np.empty((n, K), dtype=np.float64)
    all_vit_idx   = np.empty((n, K), dtype=np.int64)
    all_vit_vals  = np.empty((n, K), dtype=np.float64)

    # Coordinates are the same for every image; only values differ
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
        all_vit_vals[i] = vals.astype(np.float64)

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
    out_path: str | None = None,
    seed: int = 0,
) -> dict:
    """
    Sample n images from the MRF and observe a different random set of
    n_observed pixels in each image.

    Parameters
    ----------
    alf : (n1, n2, d) unary log-potentials
    bet : scalar coupling
    n_observed : number of pixels to observe per image
    n : number of images to generate
    out_path : if given, save output to this .npz path
    seed : RNG seed (used for both the MRF sampler and pixel selection)

    Returns
    -------
    dict with same keys as generate_fixed_observations
    """
    n1, n2 = alf.shape[:2]
    n_pixels = n1 * n2

    if not (0 < n_observed <= n_pixels):
        raise ValueError(
            f"n_observed={n_observed} must be in [1, {n_pixels}] for a ({n1},{n2}) grid."
        )

    images = generate_samples(alf, bet, n, seed=seed, dtype=np.uint8)  # (N, n1, n2)

    all_obs_rows  = np.empty((n, n_observed), dtype=np.int32)
    all_obs_cols  = np.empty((n, n_observed), dtype=np.int32)
    all_obs_vals  = np.empty((n, n_observed), dtype=np.float64)
    all_vit_idx   = np.empty((n, n_observed), dtype=np.int64)
    all_vit_vals  = np.empty((n, n_observed), dtype=np.float64)

    # Use a separate RNG for pixel selection so the MRF seed is unaffected
    rng = np.random.default_rng(seed + 1)
    rng_norm = np.random.default_rng(seed + 2)
    all_flat = np.arange(n_pixels, dtype=np.int64)

    for i, img in enumerate(images):
        chosen = rng.choice(all_flat, size=n_observed, replace=False)
        chosen.sort()  # deterministic ordering: row-major

        rows = (chosen // n2).astype(np.int32)
        cols = (chosen %  n2).astype(np.int32)
        vals = img[rows, cols].astype(np.float64) + rng_norm.normal(loc=0, scale=sigma, size=n_observed)

        all_obs_rows[i] = rows
        all_obs_cols[i] = cols
        all_obs_vals[i] = vals
        all_vit_idx[i]  = chosen
        all_vit_vals[i] = vals.astype(np.float64)

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


# ── Convenience: reconstruct obs_dict list from a saved .npz ─────────────────
def load_obs_dicts(npz_path: str) -> list[dict]:
    """
    Load a dataset saved by either function above and return a list of
    observation dicts suitable for exact_marginals.marginals(observations=...).

    Returns
    -------
    list of N dicts, each {(row, col): value}
    """
    data = np.load(npz_path)
    rows = data["obs_rows"]   # (N, K)
    cols = data["obs_cols"]   # (N, K)
    vals = data["obs_vals"]   # (N, K)
    return [
        {(int(r), int(c)): int(v) for r, c, v in zip(rows[i], cols[i], vals[i])}
        for i in range(rows.shape[0])
    ]


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n1, n2, d = 12, 24, 2
    alf = np.zeros((n1, n2, d), dtype=float)
    bet = 0.7
    sigma = 0.5

    # Function 1: observe all pixels in columns 6 and 18
    fixed_pixels = [(r, c) for c in [6, 18] for r in range(n1)]
    d1 = generate_fixed_observations(
        alf, bet, fixed_pixels, n=500, sigma=sigma, out_path="data/gauss_exp/obs_A_wells.npz", seed=280
    )
    print("Fixed obs — images:", d1["images"].shape,
          "vit_indices:", d1["vit_indices"].shape)

    # Function 2: observe 30 random pixels per image
    d2 = generate_random_observations(
        alf, bet, n_observed=30, n=500, sigma=sigma, out_path="data/gauss_exp/obs_A_random.npz", seed=50
    )
    print("Random obs — images:", d2["images"].shape,
          "vit_indices:", d2["vit_indices"].shape)
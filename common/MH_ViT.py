# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:14:12 2026

@author: brinc
"""

import numpy as np
from scipy.stats import norm
from ViT_sampling_rectangle import *

def proposal_gauss(model, indices, probs, probs_marginal, rng, order, patch_size, image_size):
    unif = rng.random(len(indices))
    x_S = (unif < probs).astype(int)
    mask = unif < probs
    S_contrib = np.sum(np.log(probs_marginal[mask])) + np.sum(np.log(1 - probs_marginal[~mask]))
    prop_img, cond_contrib, uncond_contrib = generate_image(model, patch_size, image_size,
                                                            condition_indices=indices, condition_values=x_S,
                                                            generation_order=order, return_uncond=True)
    ell_contrib = uncond_contrib - cond_contrib - S_contrib
    return prop_img, ell_contrib, uncond_contrib

def proposal_sum(model, indices, d, rng, order, patch_size, image_size):
    n = len(indices)
    x_S = np.zeros(n, dtype=int)
    x_S[rng.choice(n, size=int(d), replace=False)] = 1
    prop_img, cond_contrib, uncond_contrib = generate_image(model, patch_size, image_size,
                                                            condition_indices=indices, condition_values=x_S,
                                                            generation_order=order, return_uncond=True)
    ell_contrib = uncond_contrib - cond_contrib
    return prop_img, ell_contrib, uncond_contrib

def proposal_exact(model, indices, x_S, order, patch_size, image_size):
    prop_img, cond_contrib, uncond_contrib = generate_image(model, patch_size, image_size,
                                                            condition_indices=indices, condition_values=x_S,
                                                            generation_order=order, return_uncond=True)
    ell_contrib = uncond_contrib - cond_contrib
    return prop_img, ell_contrib, uncond_contrib

def MH_noisy_data(model, indices, data, sigma, probmap, 
                  patch_size, image_size, n_samples, burnin=0, seed=0, order="raster", verbose=False):
    rng = np.random.default_rng(seed)
    images = np.zeros((n_samples, image_size[0], image_size[1]))
    logliks = np.zeros(n_samples)
    
    probs = np.zeros(len(indices))
    probs_marginal = np.zeros(len(indices))
    for i, idx in enumerate(indices):
        p_1 = norm.pdf(data[i], 1, sigma)*probmap[int(idx)]
        p_0 = norm.pdf(data[i], 0, sigma)*(1-probmap[int(idx)])
        probs[i] = p_1/(p_1 + p_0)
        probs_marginal[i] = probmap[int(idx)]
    
    images[0], ell_current, logliks[0] = proposal_gauss(model, indices, probs, probs_marginal, rng, order, patch_size, image_size)
    
    n_accepted = 0
    
    if verbose:
        print("Starting gaussian MH...")
    
    for i in range(1, n_samples):
        img_prop, ell_prop, loglik_prop = proposal_gauss(model, indices, probs, probs_marginal, rng, order, patch_size, image_size)
        ell = ell_prop - ell_current
        unif = rng.random()
        if np.log(unif) < ell:
            images[i] = img_prop
            logliks[i] = loglik_prop
            ell_current = ell_prop
            n_accepted += 1
        else:
            images[i] = images[i-1]
            logliks[i] = logliks[i-1]
        if verbose and i%100==0:
            print(f"{i} completed. Acc.prob. so far: {n_accepted/i}.")
    
    alpha = n_accepted/(n_samples - 1)
    
    if verbose:
        print(f"Finished. Acc.prob. {alpha}; {n_accepted} of {n_samples-1} accepted.")
    
    return images[burnin:], logliks[burnin:], alpha

def MH_summed_data(model, indices, data, 
                  patch_size, image_size, n_samples, burnin=0, seed=0, order="raster", verbose=False):
    rng = np.random.default_rng(seed)
    images = np.zeros((n_samples, image_size[0], image_size[1]))
    logliks = np.zeros(n_samples)
    
    images[0], ell_current, logliks[0] = proposal_sum(model, indices, data, rng, order, patch_size, image_size)
    
    n_accepted = 0
    
    if verbose:
        print("Starting summed MH...")
    
    for i in range(1, n_samples):
        img_prop, ell_prop, loglik_prop = proposal_sum(model, indices, data, rng, order, patch_size, image_size)
        ell = ell_prop - ell_current
        unif = rng.random()
        if np.log(unif) < ell:
            images[i] = img_prop
            logliks[i] = loglik_prop
            ell_current = ell_prop
            n_accepted += 1
        else:
            images[i] = images[i-1]
            logliks[i] = logliks[i-1]
        if verbose and i%100==0:
            print(f"{i} completed. Acc.prob. so far: {n_accepted/i}.")
    
    alpha = n_accepted/(n_samples - 1)
    
    if verbose:
        print(f"Finished. Acc.prob. {alpha}; {n_accepted} of {n_samples-1} accepted.")
    
    return images[burnin:], logliks[burnin:], alpha

def MH_exact_data(model, indices, data, 
                  patch_size, image_size, n_samples, burnin=0, seed=0, order="raster", verbose=False):
    rng = np.random.default_rng(seed)
    images = np.zeros((n_samples, image_size[0], image_size[1]))
    logliks = np.zeros(n_samples)
    
    images[0], ell_current, logliks[0] = proposal_exact(model, indices, data, order, patch_size, image_size)
    
    n_accepted = 0
    
    if verbose:
        print("Starting exact MH...")
    
    for i in range(1, n_samples):
        img_prop, ell_prop, loglik_prop = proposal_exact(model, indices, data, order, patch_size, image_size)
        ell = ell_prop - ell_current
        unif = rng.random()
        if np.log(unif) < ell:
            images[i] = img_prop
            logliks[i] = loglik_prop
            ell_current = ell_prop
            n_accepted += 1
        else:
            images[i] = images[i-1]
            logliks[i] = logliks[i-1]
        if verbose and i%100==0:
            print(f"{i} completed. Acc.prob. so far: {n_accepted/i}.")
    
    alpha = n_accepted/(n_samples - 1)
    
    if verbose:
        print(f"Finished. Acc.prob. {alpha}; {n_accepted} of {n_samples-1} accepted.")
    
    return images[burnin:], logliks[burnin:], alpha

def make_observations_vertical(image, line, obs_model="exact", sigma=None, rng=None):
    """
    Observe pixels along a vertical line (fixed column).
    image: 2D array, shape (H, W), indexed as image[row, col].
    line: column index to observe.
    Returns flat row-major indices and observations.
    """
    assert image.ndim == 2
    assert obs_model in ("exact", "sum", "gauss")
    H, W = image.shape
    assert 0 <= line < W, f"line={line} out of bounds for width {W}"
    if obs_model == "gauss":
        assert sigma is not None and sigma > 0 and rng is not None

    rows = np.arange(H)
    indices = rows * W + line          # flat row-major: row*W + col
    truth = image[rows, line]          # image[row, col]

    if obs_model == "exact":
        return indices, truth
    elif obs_model == "sum":
        return indices, int(np.sum(truth))
    elif obs_model == "gauss":
        return indices, truth + rng.standard_normal(H) * sigma


def make_observations_horizontal(image, line, obs_model="exact", sigma=None, rng=None):
    """
    Observe pixels along a horizontal line (fixed row).
    image: 2D array, shape (H, W), indexed as image[row, col].
    line: row index to observe.
    Returns flat row-major indices and observations.
    """
    assert image.ndim == 2
    assert obs_model in ("exact", "sum", "gauss")
    H, W = image.shape
    assert 0 <= line < H, f"line={line} out of bounds for height {H}"
    if obs_model == "gauss":
        assert sigma is not None and sigma > 0 and rng is not None

    cols = np.arange(W)
    indices = line * W + cols          # flat row-major: row*W + col
    truth = image[line, cols]          # image[row, col]

    if obs_model == "exact":
        return indices, truth
    elif obs_model == "sum":
        return indices, int(np.sum(truth))
    elif obs_model == "gauss":
        return indices, truth + rng.standard_normal(W) * sigma


def make_observations_diagonal_vertical(image, top, bottom, obs_model="exact", sigma=None, rng=None):
    """
    Observe pixels along a diagonal crossing the full height.
    top: column at row 0, bottom: column at row H-1.
    image: 2D array, shape (H, W), indexed as image[row, col].
    Returns flat row-major indices and observations.
    """
    assert image.ndim == 2
    assert obs_model in ("exact", "sum", "gauss")
    H, W = image.shape
    assert 0 <= top < W,    f"top={top} out of bounds for width {W}"
    assert 0 <= bottom < W, f"bottom={bottom} out of bounds for width {W}"
    if obs_model == "gauss":
        assert sigma is not None and sigma > 0 and rng is not None

    rows = np.arange(H)
    cols = np.round(np.linspace(top, bottom, H)).astype(int)
    indices = rows * W + cols          # flat row-major
    truth = image[rows, cols]          # image[row, col]

    if obs_model == "exact":
        return indices, truth
    elif obs_model == "sum":
        return indices, int(np.sum(truth))
    elif obs_model == "gauss":
        return indices, truth + rng.standard_normal(H) * sigma


def make_observations_diagonal_horizontal(image, top, bottom, obs_model="exact", sigma=None, rng=None):
    """
    Observe pixels along a diagonal crossing the full width.
    top: row at col 0, bottom: row at col W-1.
    image: 2D array, shape (H, W), indexed as image[row, col].
    Returns flat row-major indices and observations.
    """
    assert image.ndim == 2
    assert obs_model in ("exact", "sum", "gauss")
    H, W = image.shape
    assert 0 <= top < H,    f"top={top} out of bounds for height {H}"
    assert 0 <= bottom < H, f"bottom={bottom} out of bounds for height {H}"
    if obs_model == "gauss":
        assert sigma is not None and sigma > 0 and rng is not None

    cols = np.arange(W)
    rows = np.round(np.linspace(top, bottom, W)).astype(int)
    indices = rows * W + cols          # flat row-major
    truth = image[rows, cols]          # image[row, col]

    if obs_model == "exact":
        return indices, truth
    elif obs_model == "sum":
        return indices, int(np.sum(truth))
    elif obs_model == "gauss":
        return indices, truth + rng.standard_normal(W) * sigma
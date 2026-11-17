# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from MH_ViT import *

IMAGE_SIZE = (12, 24)  # (H, W) — adjust if needed
PATCH_SIZE = 2
LINE = 12              # column to observe
N_SAMPLES = 500
BURNIN = 0
SEED = 42
ORDER = "manhattan"


def _probmap_flat(image_size):
    H, W = image_size
    return np.full(H * W, 0.5)


def make_observations_random(image, n_obs, obs_model="exact", sigma=None, rng=None, seed=42):
    """
    Observe n_obs randomly chosen pixels from image.
    Returns flat row-major indices and observations.
    """
    assert image.ndim == 2
    assert obs_model in ("exact", "sum", "gauss")
    H, W = image.shape
    assert 0 < n_obs <= H * W
    if obs_model == "gauss":
        assert sigma is not None and sigma > 0

    if rng is None:
        rng = np.random.default_rng(seed)

    indices = rng.choice(H * W, size=n_obs, replace=False)
    indices = np.sort(indices)
    rows, cols = np.divmod(indices, W)
    truth = image[rows, cols]

    if obs_model == "exact":
        return indices, truth
    elif obs_model == "sum":
        return indices, int(np.sum(truth))
    elif obs_model == "gauss":
        return indices, truth + rng.standard_normal(n_obs) * sigma


def _plot_results_line(images, logliks, truth, title, line):
    """Plot helper for line observations."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(title)

    axes[0].plot(logliks)
    axes[0].set_title("ViT log-likelihood trace")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("log p(x)")

    axes[1].imshow(truth, cmap="gray", vmin=0, vmax=1)
    axes[1].axvline(x=line, color="red", linewidth=1.5, label="Observed column")
    axes[1].set_title("Ground truth")
    axes[1].legend(fontsize=7)

    mean_img = images.mean(axis=0)
    axes[2].imshow(mean_img, cmap="gray", vmin=0, vmax=1)
    axes[2].axvline(x=line, color="red", linewidth=1.5)
    axes[2].set_title("Posterior mean")

    axes[3].imshow(np.abs(mean_img - truth), cmap="hot", vmin=0, vmax=1)
    axes[3].axvline(x=line, color="blue", linewidth=1.5)
    axes[3].set_title("|Mean - Truth|")

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150)
    plt.show()


def _plot_results_random(images, logliks, truth, indices, title):
    """Plot helper for random observations — marks observed pixels as dots."""
    H, W = truth.shape
    rows, cols = np.divmod(indices, W)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(title)

    axes[0].plot(logliks)
    axes[0].set_title("ViT log-likelihood trace")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("log p(x)")

    axes[1].imshow(truth, cmap="gray", vmin=0, vmax=1)
    axes[1].scatter(cols, rows, c="red", s=8, label="Observed")
    axes[1].set_title("Ground truth")
    axes[1].legend(fontsize=7)

    mean_img = images.mean(axis=0)
    axes[2].imshow(mean_img, cmap="gray", vmin=0, vmax=1)
    axes[2].scatter(cols, rows, c="red", s=8)
    axes[2].set_title("Posterior mean")

    axes[3].imshow(np.abs(mean_img - truth), cmap="hot", vmin=0, vmax=1)
    axes[3].scatter(cols, rows, c="blue", s=8)
    axes[3].set_title("|Mean - Truth|")

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150)
    plt.show()


def test_exact(image, model):
    """MH with exact pixel observations along a vertical line."""
    indices, data = make_observations_vertical(image, LINE, obs_model="exact")
    images, logliks, alpha = MH_exact_data(
        model, indices, data,
        patch_size=PATCH_SIZE, image_size=IMAGE_SIZE,
        n_samples=N_SAMPLES, burnin=BURNIN, seed=SEED, order=ORDER, verbose=True
    )
    print(f"[Exact] Acceptance rate: {alpha:.3f}")
    _plot_results_line(images, logliks, image, "Exact observations", LINE)
    return images, logliks, alpha


def test_gauss(image, model, sigma=0.3):
    """MH with Gaussian-noisy pixel observations along a vertical line."""
    rng = np.random.default_rng(SEED)
    indices, data = make_observations_vertical(image, LINE, obs_model="gauss", sigma=sigma, rng=rng)
    probmap = _probmap_flat(IMAGE_SIZE)
    images, logliks, alpha = MH_noisy_data(
        model, indices, data, sigma, probmap,
        patch_size=PATCH_SIZE, image_size=IMAGE_SIZE,
        n_samples=N_SAMPLES, burnin=BURNIN, seed=SEED, order=ORDER, verbose=True
    )
    print(f"[Gauss sigma={sigma}] Acceptance rate: {alpha:.3f}")
    _plot_results_line(images, logliks, image, f"Gaussian observations sigma={sigma}", LINE)
    return images, logliks, alpha


def test_sum(image, model):
    """MH with summed pixel observations along a vertical line."""
    indices, data = make_observations_vertical(image, LINE, obs_model="sum")
    images, logliks, alpha = MH_summed_data(
        model, indices, data,
        patch_size=PATCH_SIZE, image_size=IMAGE_SIZE,
        n_samples=N_SAMPLES, burnin=BURNIN, seed=SEED, order=ORDER, verbose=True
    )
    print(f"[Sum={data}] Acceptance rate: {alpha:.3f}")
    _plot_results_line(images, logliks, image, f"Summed observations d={data}", LINE)
    return images, logliks, alpha


def test_exact_random(image, model, n_obs=20):
    """MH with exact observations at n_obs randomly chosen pixels."""
    indices, data = make_observations_random(image, n_obs, obs_model="exact", seed=SEED)
    images, logliks, alpha = MH_exact_data(
        model, indices, data,
        patch_size=PATCH_SIZE, image_size=IMAGE_SIZE,
        n_samples=N_SAMPLES, burnin=BURNIN, seed=SEED, order=ORDER, verbose=True
    )
    print(f"[Exact random n={n_obs}] Acceptance rate: {alpha:.3f}")
    _plot_results_random(images, logliks, image, indices, f"Exact random n={n_obs}")
    return images, logliks, alpha


def test_gauss_random(image, model, n_obs=20, sigma=0.3):
    """MH with Gaussian-noisy observations at n_obs randomly chosen pixels."""
    rng = np.random.default_rng(SEED)
    indices, data = make_observations_random(image, n_obs, obs_model="gauss", sigma=sigma, rng=rng)
    probmap = _probmap_flat(IMAGE_SIZE)
    images, logliks, alpha = MH_noisy_data(
        model, indices, data, sigma, probmap,
        patch_size=PATCH_SIZE, image_size=IMAGE_SIZE,
        n_samples=N_SAMPLES, burnin=BURNIN, seed=SEED, order=ORDER, verbose=True
    )
    print(f"[Gauss random n={n_obs} sigma={sigma}] Acceptance rate: {alpha:.3f}")
    _plot_results_random(images, logliks, image, indices, f"Gaussian random n={n_obs} sigma={sigma}")
    return images, logliks, alpha


def test_sum_random(image, model, n_obs=20):
    """MH with summed observations over n_obs randomly chosen pixels."""
    indices, data = make_observations_random(image, n_obs, obs_model="sum", seed=SEED)
    images, logliks, alpha = MH_summed_data(
        model, indices, data,
        patch_size=PATCH_SIZE, image_size=IMAGE_SIZE,
        n_samples=N_SAMPLES, burnin=BURNIN, seed=SEED, order=ORDER, verbose=True
    )
    print(f"[Sum random n={n_obs} d={data}] Acceptance rate: {alpha:.3f}")
    _plot_results_random(images, logliks, image, indices, f"Summed random n={n_obs} d={data}")
    return images, logliks, alpha

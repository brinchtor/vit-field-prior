"""
plot_fluvial.py

Produces qualitative and quantitative figures for the fluvial 64x64 experiment.

Required data files (relative to script location, or override paths below):
  data/fluvial_binary_64x64.h5                 – full dataset (test set = last 1000)
  data/unconditional_samples_fluvial_64.npz    – 1000 unconditional ViT samples
  data/unconditional_samples_fluvial_64.npz    – also reused for conditional single
  data/conditional_samples_fluvial_64.npz      – single-image conditional run
                                                  (keys: images, log_lik_cond,
                                                   log_lik_uncond, observed_image,
                                                   condition_cols)
  data/fluvial64_testset_experiment.npz        – full testset conditional run
                                                  (keys: observed_images,
                                                   mean_unweighted, mean_weighted,
                                                   ess, condition_cols)

Outputs:  figures_fluvial/<name>.png
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py

os.makedirs("figures_fluvial_0211", exist_ok=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
parent_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir     = os.path.join(parent_dir, "data")

h5_path      = os.path.join(data_dir, "fluvial_binary_64x64.h5")
uncond_path  = os.path.join(data_dir, "rowonly_bigger_epoch_1000_n1000.npz")
uncond_path_revraster  = os.path.join(data_dir, "rowonly_bigger_revraster_epoch_1000_n1000.npz")
uncond_path_random  = os.path.join(data_dir, "unconditional_samples_fluvial_random_64.npz")
cond1_path   = os.path.join(data_dir, "conditional_samples_fluvial_64.npz")
testset_path = os.path.join(data_dir, "fluvial64_testset_experiment.npz")

# ── Helper ────────────────────────────────────────────────────────────────────
def savefig(name):
    path = f"figures_fluvial_0211/{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  3×3 grid of unconditional ViT samples
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 1: unconditional samples grid …")
uncond = np.load(uncond_path)
uncond_imgs = uncond["images"]          # (1000, H, W)
uncond_revraster = np.load(uncond_path_revraster)
uncond_imgs_revraster = uncond_revraster["images"]          # (1000, H, W)

fig, axes = plt.subplots(3, 3, figsize=(5, 5))
for ax, img in zip(axes.ravel(), uncond_imgs[:9]):
    ax.imshow(img, cmap="gray", interpolation="none", origin="upper", vmin=0, vmax=1)
    ax.axis("off")
plt.tight_layout()
savefig("uncond_samples_grid")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  3×3 grid of real test images
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 2: real test images grid …")
with h5py.File(h5_path, "r") as f:
    real_imgs = f["images"][-1000:-991]   # first 9 of held-out test set

fig, axes = plt.subplots(3, 3, figsize=(5, 5))
for ax, img in zip(axes.ravel(), real_imgs):
    ax.imshow(img, cmap="gray", interpolation="none", origin="upper", vmin=0, vmax=1)
    ax.axis("off")
plt.tight_layout()
savefig("real_test_grid")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Unconditional mean  vs  real mean  vs  difference
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 3: mean comparison …")
# load full test set for real mean
with h5py.File(h5_path, "r") as f:
    test_imgs_full = f["images"][-1000:].astype(np.float32)

real_mean   = test_imgs_full.mean(axis=0)           # (H, W)
uncond_mean = uncond_imgs.mean(axis=0)              # (H, W)
uncond_mean_revraster = uncond_imgs_revraster.mean(axis=0)              # (H, W)
diff        = uncond_mean - real_mean               # positive → ViT higher
diff2       = uncond_mean_revraster - real_mean               # positive → ViT higher

# diverging colormap centred at zero
vabs = np.abs(diff).max()
norm_div = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)

im0 = axes[0].imshow(uncond_mean, cmap="gray", interpolation="none",
                     origin="upper", vmin=0, vmax=1)
axes[0].set_title("Unconditional ViT mean", fontsize=9)
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(real_mean, cmap="gray", interpolation="none",
                     origin="upper", vmin=0, vmax=1)
axes[1].set_title("Real test-set mean", fontsize=9)
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(diff, cmap="RdBu_r", norm=norm_div,
                     interpolation="none", origin="upper")
axes[2].set_title("ViT mean − Real mean", fontsize=9)
axes[2].axis("off")
cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
cbar2.set_label("Difference", fontsize=8)

vabs = np.abs(diff2).max()
norm_div = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

im3 = axes[3].imshow(diff2, cmap="RdBu_r", norm=norm_div,
                     interpolation="none", origin="upper")
axes[3].set_title("ViT mean − Real mean (reverse raster)", fontsize=9)
axes[3].axis("off")
cbar3 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
cbar3.set_label("Difference", fontsize=8)

savefig("mean_comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Single-image conditional: observed image, unweighted mean, IS-weighted mean
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 4: conditional single-image …")
try:
    cond1 = np.load(cond1_path)
    obs_img    = cond1["observed_image"]        # (H, W)
    cond_imgs  = cond1["images"]                # (N, H, W)
    log_cond   = cond1["log_lik_cond"]
    log_uncond = cond1["log_lik_uncond"]
    cond_cols  = cond1["condition_cols"]

    log_w = log_uncond - log_cond
    log_w -= log_w.max()
    w = np.exp(log_w); w /= w.sum()
    ess_single = 1.0 / (w ** 2).sum()

    mean_unw = cond_imgs.mean(axis=0)
    mean_w   = (w[:, None, None] * cond_imgs).sum(axis=0)

    # build observed-pixel overlay: grey=unobserved, true value elsewhere
    H_img, W_img = obs_img.shape
    obs_display = np.full((H_img, W_img), np.nan)
    for col in cond_cols:
        obs_display[:, col] = obs_img[:, col]

    fig, axes = plt.subplots(1, 3, figsize=(8, 3), constrained_layout=True)
    panels = [
        (obs_img,   f"Test image\n(columns {list(cond_cols)} observed)"),
        (mean_unw,  "Unweighted posterior mean"),
        (mean_w,    f"IS-weighted posterior mean\n(ESS = {ess_single:.0f})"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray", interpolation="none",
                  origin="upper", vmin=0, vmax=1)
        # highlight observed columns with a semi-transparent red overlay
        col_mask = np.zeros((H_img, W_img, 4), dtype=np.float32)
        for col in cond_cols:
            col_mask[:, col, 0] = 1.0   # R
            col_mask[:, col, 3] = 0.35  # alpha
        ax.imshow(col_mask, interpolation="none", origin="upper")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    savefig("cond_single_image")
    print(f"  ESS = {ess_single:.1f}/{len(log_cond)}")
except FileNotFoundError:
    print(f"  WARNING: {cond1_path} not found – skipping Figure 4.")


# ═══════════════════════════════════════════════════════════════════════════════
# 5–7.  Quantitative plots from generate_conditional_testset.py output
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading testset experiment …")
try:
    ts = np.load(testset_path)
    obs_imgs   = ts["observed_images"]    # (N_TEST, H, W)
    mean_unw_ts = ts["mean_unweighted"]   # (N_TEST, H, W)
    mean_w_ts   = ts["mean_weighted"]     # (N_TEST, H, W)
    ess_ts      = ts["ess"]               # (N_TEST,)
    N_TEST = len(obs_imgs)
    print(f"  Loaded {N_TEST} test results.")

    # ── MSE vs true image ────────────────────────────────────────────────────
    # "true posterior" is unavailable; we compare against ground-truth pixel values.
    target   = obs_imgs                        # (N_TEST, H, W)
    MSE_unw  = ((mean_unw_ts - target) ** 2).mean(axis=(1, 2))
    MSE_w    = ((mean_w_ts   - target) ** 2).mean(axis=(1, 2))

    # ── Figure 5: UW vs W scatter coloured by ESS ────────────────────────────
    print("Figure 5: MSE scatter UW vs W …")
    lo = min(MSE_unw.min(), MSE_w.min()) - 5e-5
    hi = max(MSE_unw.max(), MSE_w.max()) + 5e-5
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, alpha=0.7)
    sc = ax.scatter(MSE_unw, MSE_w, s=12, c=ess_ts, cmap="viridis", alpha=0.75)
    cbar = plt.colorbar(sc, ax=ax, label="ESS")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Unweighted MSE (vs truth)")
    ax.set_ylabel("IS-weighted MSE (vs truth)")
    ax.set_title("MSE scatter: unweighted vs IS-weighted", fontsize=9)
    ax.grid(True, linewidth=0.4)
    plt.tight_layout()
    savefig("mse_scatter_uw_vs_w")

    n_better = np.sum(MSE_w < MSE_unw)
    print(f"  Mean MSE UW={MSE_unw.mean():.6f}  W={MSE_w.mean():.6f}  "
          f"IS better in {100*n_better/N_TEST:.1f}% of cases")

    # ── Figure 6: Spatial error maps ─────────────────────────────────────────
    print("Figure 6: spatial error maps …")
    err_unw = ((mean_unw_ts - target) ** 2).mean(axis=0)  # (H, W)
    err_w   = ((mean_w_ts   - target) ** 2).mean(axis=0)

    vmax = max(err_unw.max(), err_w.max())
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    for ax, err, lbl in zip(axes, [err_unw, err_w], ["Unweighted", "IS-weighted"]):
        im = ax.imshow(err, cmap="hot", interpolation="none",
                       origin="upper", vmin=0, vmax=vmax)
        ax.set_title(lbl, fontsize=9)
        ax.axis("off")
    cbar = plt.colorbar(im, ax=axes.ravel().tolist(),
                        orientation="horizontal", shrink=0.8,
                        aspect=30, pad=0.08)
    cbar.set_label("Mean squared error (vs truth)", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    savefig("spatial_error")

    # ── Figure 7: ESS histogram ───────────────────────────────────────────────
    print("Figure 7: ESS histogram …")
    N_samples_ts = int(round(ess_ts.max()))   # N used in the testset run (approx)
    # try to read from file metadata, fall back to 5000 default
    # (generate_conditional_testset uses N_SAMPLES=5000 by default)
    N_SAMPLES_TS = 1000

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(ess_ts, bins=10, density=True, color="darkorange",
            edgecolor="white", linewidth=0.5)
    ax.axvline(N_SAMPLES_TS, color="black", linewidth=0.9,
               linestyle="--", alpha=0.7, label=f"$N = {N_SAMPLES_TS}$")
    ax.set_xlabel("ESS"); ax.set_ylabel("Density")
    ax.set_title("Effective sample size across test images", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4)
    plt.tight_layout()
    savefig("ess_histogram")

    print(f"  ESS  mean={ess_ts.mean():.1f}  "
          f"min={ess_ts.min():.1f}  max={ess_ts.max():.1f}")

except FileNotFoundError:
    print(f"  WARNING: {testset_path} not found – skipping Figures 5–7.")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Mean comparison: random unconditional ViT vs real mean
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 8: mean comparison (random unconditional) …")
try:
    uncond_random = np.load(uncond_path_random)
    uncond_random_imgs = uncond_random["images"]        # (N, H, W)

    uncond_random_mean = uncond_random_imgs.mean(axis=0)
    diff_random        = uncond_random_mean - real_mean

    vabs_r = np.abs(diff_random).max()
    norm_div_r = mcolors.TwoSlopeNorm(vmin=-vabs_r, vcenter=0, vmax=vabs_r)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), constrained_layout=True)

    im0 = axes[0].imshow(uncond_random_mean, cmap="gray", interpolation="none",
                         origin="upper", vmin=0, vmax=1)
    axes[0].set_title("Unconditional ViT mean (random)", fontsize=9)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(real_mean, cmap="gray", interpolation="none",
                         origin="upper", vmin=0, vmax=1)
    axes[1].set_title("Real test-set mean", fontsize=9)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff_random, cmap="RdBu_r", norm=norm_div_r,
                         interpolation="none", origin="upper")
    axes[2].set_title("ViT mean (random) − Real mean", fontsize=9)
    axes[2].axis("off")
    cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.set_label("Difference", fontsize=8)

    savefig("mean_comparison_random")
except FileNotFoundError:
    print(f"  WARNING: {uncond_path_random} not found – skipping Figure 8.")


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: count connected components of value 1 per image stack
# ═══════════════════════════════════════════════════════════════════════════════
def count_clusters(imgs):
    """Return array of cluster counts (connected components with value 1) per image.
    Uses 4-connectivity (scipy default cross-shaped structuring element)."""
    from scipy import ndimage
    counts = np.empty(len(imgs), dtype=int)
    for i, img in enumerate(imgs):
        binary = img.astype(bool)
        _, n = ndimage.label(binary)
        counts[i] = n
    return counts


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Cluster-count histogram: real test set vs unconditional ViT
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 9: cluster-count histogram (real vs unconditional ViT) …")
real_clusters   = count_clusters(test_imgs_full.astype(np.float32))
uncond_clusters = count_clusters(uncond_imgs)

all_counts = np.concatenate([real_clusters, uncond_clusters])
bin_edges  = np.arange(all_counts.min(), all_counts.max() + 2) - 0.5

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.hist(real_clusters,   bins=bin_edges, density=True, alpha=0.6,
        color="steelblue",  edgecolor="white", linewidth=0.4, label="Real test set")
ax.hist(uncond_clusters, bins=bin_edges, density=True, alpha=0.6,
        color="darkorange", edgecolor="white", linewidth=0.4, label="Unconditional ViT")
ax.set_xlabel("Number of clusters (pixel value 1)")
ax.set_ylabel("Density")
ax.legend(fontsize=8)
ax.grid(True, linewidth=0.4)
plt.tight_layout()
savefig("cluster_histogram_uncond")

print(f"  Real clusters:   mean={real_clusters.mean():.1f}  "
      f"median={np.median(real_clusters):.0f}")
print(f"  ViT clusters:    mean={uncond_clusters.mean():.1f}  "
      f"median={np.median(uncond_clusters):.0f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  Cluster-count histogram: real test set vs random unconditional ViT
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 10: cluster-count histogram (real vs random unconditional ViT) …")
try:
    uncond_random_clusters = count_clusters(uncond_random_imgs)

    all_counts_r = np.concatenate([real_clusters, uncond_random_clusters])
    bin_edges_r  = np.arange(all_counts_r.min(), all_counts_r.max() + 2) - 0.5

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(real_clusters,          bins=bin_edges_r, density=True, alpha=0.6,
            color="steelblue",  edgecolor="white", linewidth=0.4, label="Real test set")
    ax.hist(uncond_random_clusters, bins=bin_edges_r, density=True, alpha=0.6,
            color="mediumpurple", edgecolor="white", linewidth=0.4,
            label="Unconditional ViT (random)")
    ax.set_xlabel("Number of clusters (pixel value 1)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4)
    plt.tight_layout()
    savefig("cluster_histogram_uncond_random")

    print(f"  ViT random clusters: mean={uncond_random_clusters.mean():.1f}  "
          f"median={np.median(uncond_random_clusters):.0f}")
except NameError:
    print("  WARNING: random unconditional images not loaded – skipping Figure 10.")


print("\nDone.  All figures saved to figures_fluvial/")
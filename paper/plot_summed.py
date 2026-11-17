#!/usr/bin/env python
# coding: utf-8
"""
Summed-observation experiment figures (one well, models A-D).

Reads, per model M, from data/summed_exp/:
    obs_{M}_well.npz    images, obs_col, obs_sum, well_rows, well_cols, vit_indices
    exact_{M}_well.npz  marginals [N,H,W,2]  (analytical E[x|d]), logZ
    mrf_{M}_well.npz    images   [N, S, H, W]  (exact conditional MRF samples)
    vit_{M}_well.npz    samples, log_lik_cond, log_lik_uncond, log_r

IS weight for the summed model (per image, self-normalised over the S samples):
    log w = log_lik_uncond - log_lik_cond - log_r
For models A/C the marginals are flat 0.5, so log_r is constant per image and
cancels in the normalisation; for B/D it varies and matters.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures_summed", exist_ok=True)

MODELS = ["A", "B", "C", "D"]
base   = "data/summed_exp"

# example image index to display per model (Fig 14)
i_shows = [0, 1, 2, 0]

for count, model in enumerate(MODELS):
    title_tag = f"Model {model}, summed (one well)"
    print(f"\n=== {title_tag} ===")

    data     = np.load(f"{base}/obs_{model}_well.npz")
    exact    = np.load(f"{base}/exact_{model}_well.npz")
    mrf_data = np.load(f"{base}/mrf_{model}_well.npz")
    npz      = np.load(f"{base}/vit_{model}_well.npz")

    true_images = data["images"]
    marginals   = exact["marginals"]
    mrf_imgs    = mrf_data["images"]

    imgs           = npz["samples"]
    log_lik_cond   = npz["log_lik_cond"]
    log_lik_uncond = npz["log_lik_uncond"]
    log_r          = npz["log_r"]

    # well metadata (the observation is the column total, not pixel values)
    obs_col   = int(data["obs_col"])
    obs_sum   = data["obs_sum"]      # (N,)
    well_rows = data["well_rows"]    # (L,)
    well_cols = data["well_cols"]    # (L,)

    n, N, H, W = imgs.shape
    print(f"  {n} experiments, {N} samples each, {H}x{W}; well col {obs_col}")

    tag = model  # used in saved filenames

    # ── IS weights & ESS (summed: subtract log_r) ─────────────────────────────
    log_w = log_lik_uncond - log_lik_cond - log_r
    log_w -= log_w.max(axis=1, keepdims=True)
    w      = np.exp(log_w)
    w     /= w.sum(axis=1, keepdims=True)
    ess    = 1.0 / (w ** 2).sum(axis=1)
    print(np.min(ess))
    print(np.max(ess))

    # ── Posterior means ───────────────────────────────────────────────────────
    mean_unweighted = imgs.mean(axis=1)
    mean_weighted   = (w[:, :, None, None] * imgs).sum(axis=1)
    mean_mrf        = mrf_imgs.mean(axis=1)
    mean_mrf_ess    = np.stack([
        mrf_imgs[i, :max(1, int(ess[i]))].mean(axis=0) for i in range(n)
    ])

    # ── MSE ───────────────────────────────────────────────────────────────────
    target = marginals[:, :, :, 1]
    MSEs_unweighted = ((mean_unweighted - target) ** 2).mean(axis=(1, 2))
    MSEs_weighted   = ((mean_weighted   - target) ** 2).mean(axis=(1, 2))
    MSEs_mrf        = ((mean_mrf        - target) ** 2).mean(axis=(1, 2))
    MSEs_mrf_ess    = ((mean_mrf_ess    - target) ** 2).mean(axis=(1, 2))

    MMSE_u = MSEs_unweighted.mean()
    MMSE_w = MSEs_weighted.mean()
    MMSE_m = MSEs_mrf.mean()
    n_better        = np.sum(MSEs_weighted < MSEs_unweighted)
    n_worse_mrf     = np.sum(MSEs_weighted >= MSEs_mrf)
    n_worse_mrf_ess = np.sum(MSEs_weighted >= MSEs_mrf_ess)
    print(f"  Mean unweighted MSE         : {MMSE_u:.6f}")
    print(f"  Mean IS-weighted MSE        : {MMSE_w:.6f}")
    print(f"  Mean MRF MSE                : {MMSE_m:.6f}")
    print(f"  Ratio weighted/unweighted   : {MMSE_w/MMSE_u:.4f}")
    print(f"  IS better in                : {100*n_better/n:.1f}% of cases")
    print(f"  IS worse than MRF in        : {100*n_worse_mrf/n:.1f}% of cases")
    print(f"  IS worse than ESS-MRF in    : {100*n_worse_mrf_ess/n:.1f}% of cases")
    print(f"  Mean ESS                    : {ess.mean():.1f} / {N}")

    def savefig(name):
        plt.savefig(f"figures_summed/{name}_{tag}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── Fig 1: Marginals grid ─────────────────────────────────────────────────
    n_show = min(3, n)
    fig, axes = plt.subplots(n_show, 3, figsize=(7, 1.5 * n_show))
    for j, col_title in enumerate(["Exact marginal", "Unweighted mean", "IS-weighted mean"]):
        axes[0, j].set_title(col_title, fontsize=9)
    for i in range(n_show):
        axes[i, 0].imshow(marginals[i, :, :, 1], cmap="gray", interpolation="none", origin="upper")
        axes[i, 1].imshow(mean_unweighted[i],     cmap="gray", interpolation="none", origin="upper")
        axes[i, 2].imshow(mean_weighted[i],       cmap="gray", interpolation="none", origin="upper")
        for j in range(3):
            axes[i, j].axis("off")
    plt.tight_layout()
    savefig("marginals")

    # ── Fig 2: MSE histograms (all three) ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.hist(MSEs_unweighted, bins=10, alpha=0.5, label="Unweighted", density=True)
    ax.hist(MSEs_weighted,   bins=10, alpha=0.5, label="IS-weighted", density=True)
    ax.hist(MSEs_mrf,        bins=10, alpha=0.5, label="MRF", density=True)
    ax.set_xlabel("MSE"); ax.set_ylabel("Density")
    ax.legend(fontsize=8); ax.grid(True, linewidth=0.4)
    plt.tight_layout()
    savefig("mse_hist")

    # ── Fig 3: MSE histograms (ESS-MRF instead of full MRF) ───────────────────
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.hist(MSEs_unweighted, bins=10, alpha=0.5, label="Unweighted", density=True)
    ax.hist(MSEs_weighted,   bins=10, alpha=0.5, label="IS-weighted", density=True)
    ax.hist(MSEs_mrf_ess,    bins=10, alpha=0.5, label="ESS-MRF", density=True)
    ax.set_xlabel("MSE"); ax.set_ylabel("Density")
    ax.legend(fontsize=8); ax.grid(True, linewidth=0.4)
    plt.tight_layout()
    savefig("mse_hist_ess")

    # ── Fig 4: Boxplot (all three) ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bp = ax.boxplot(
        [MSEs_unweighted, MSEs_weighted, MSEs_mrf],
        labels=["Unweighted", "IS-weighted", "MRF"],
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], ["C0", "C1", "C2"]):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    ax.set_ylabel("MSE")
    ax.grid(True, linewidth=0.4)
    plt.tight_layout()
    savefig("mse_boxplot")

    # ── Fig 5: Boxplot (ESS-MRF) ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bp = ax.boxplot(
        [MSEs_unweighted, MSEs_weighted, MSEs_mrf_ess],
        labels=["Unweighted", "IS-weighted", "ESS-MRF"],
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], ["C0", "C1", "C2"]):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    ax.set_ylabel("MSE")
    ax.grid(True, linewidth=0.4)
    plt.tight_layout()
    savefig("mse_boxplot_ess")

    # ── Fig 6: MSE improvement histogram ─────────────────────────────────────
    improvement = MSEs_unweighted - MSEs_weighted
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.hist(improvement, bins=12, density=True, color="steelblue", edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("MSE reduction (unweighted $-$ weighted)")
    ax.set_ylabel("Density")
    ax.grid(True, linewidth=0.4)
    plt.tight_layout()
    savefig("mse_improvement")

    # ── Fig 7: ESS histogram ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(2, 3))
    ax.hist(ess, bins=12, density=True, color="darkorange", edgecolor="white")
    ax.axvline(N, color="black", linewidth=0.8, linestyle="--", alpha=0.6, label=f"$N={N}$")
    ax.set_xlabel("ESS"); ax.set_ylabel("Density")
    ax.grid(True, linewidth=0.4)
    plt.tight_layout()
    savefig("ess")

    # ── Fig 8: Scatter unweighted vs weighted ─────────────────────────────────
    lo = min(MSEs_weighted.min(), MSEs_unweighted.min()) - 5e-5
    hi = max(MSEs_weighted.max(), MSEs_unweighted.max()) + 5e-5
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, alpha=0.7)
    sc = ax.scatter(MSEs_unweighted, MSEs_weighted, s=12, c=ess, cmap="viridis", alpha=0.7)
    plt.colorbar(sc, ax=ax, label="ESS")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Unweighted MSE"); ax.set_ylabel("IS-weighted MSE")
    ax.grid(True, linewidth=0.4)
    plt.tight_layout()
    savefig("mse_scatter_uw_vs_w")

    # ── Fig 9: Scatter weighted vs MRF (colored by ESS) ──────────────────────
    lo = min(MSEs_mrf.min(), MSEs_weighted.min()) - 5e-5
    hi = max(MSEs_mrf.max(), MSEs_weighted.max()) + 5e-5
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, alpha=0.7)
    sc = ax.scatter(MSEs_weighted, MSEs_mrf, s=12, c=ess, cmap="viridis", alpha=0.8)
    plt.colorbar(sc, ax=ax, label="ESS")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("IS-weighted MSE"); ax.set_ylabel("MRF MSE")
    ax.grid(True, linewidth=0.4)
    plt.tight_layout()
    savefig("mse_scatter_w_vs_mrf")

    # ── Fig 10: Scatter weighted vs ESS-MRF ──────────────────────────────────
    lo = min(MSEs_mrf_ess.min(), MSEs_weighted.min()) - 5e-5
    hi = max(MSEs_mrf_ess.max(), MSEs_weighted.max()) + 5e-5
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, alpha=0.7)
    sc = ax.scatter(MSEs_weighted, MSEs_mrf_ess, s=12, c=ess, cmap="viridis", alpha=0.7)
    plt.colorbar(sc, ax=ax, label="ESS")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("IS-weighted MSE"); ax.set_ylabel("ESS-MRF MSE")
    ax.grid(True, linewidth=0.4)
    plt.tight_layout()
    savefig("mse_scatter_w_vs_mrf_ess")

    # ── Fig 11: Relative MSE ratio vs unweighted MSE ──────────────────────────
    ratio = MSEs_weighted / MSEs_unweighted
    lo_r = MSEs_unweighted.min() - 5e-5
    hi_r = MSEs_unweighted.max() + 5e-5
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.scatter(MSEs_unweighted, ratio, s=12, alpha=0.7)
    ax.set_xlim(lo_r, hi_r); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Unweighted MSE")
    ax.set_ylabel("Weighted / Unweighted MSE")
    ax.grid(True, linewidth=0.4)
    plt.tight_layout()
    savefig("mse_ratio")

    # ── Fig 12: Ratio vs ESS ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.scatter(ess, ratio, s=12, alpha=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("ESS")
    ax.set_ylabel("Weighted / Unweighted MSE")
    ax.grid(True, linewidth=0.4)
    plt.tight_layout()
    savefig("mse_ratio_vs_ess")

    # ── Fig 13: Spatial error maps ────────────────────────────────────────────
    err_unweighted = ((mean_unweighted - target) ** 2).mean(axis=0)
    err_weighted   = ((mean_weighted   - target) ** 2).mean(axis=0)
    err_mrf        = ((mean_mrf        - target) ** 2).mean(axis=0)
    vmax = max(err_unweighted.max(), err_weighted.max(), err_mrf.max())
    fig, axes = plt.subplots(1, 2, figsize=(3, 1.5), constrained_layout=True)
    for ax, err, lbl in zip(axes,
                            [err_unweighted, err_weighted],
                            ["Unweighted", "IS-weighted"]):
        im = ax.imshow(err, cmap="hot", interpolation="none",
                       origin="upper", vmin=0, vmax=vmax)
        ax.set_title(lbl, fontsize=9); ax.axis("off")
        # mark the well column
        ax.axvline(obs_col, color="cyan", linewidth=0.6, alpha=0.7)
    cbar = plt.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal",
                        shrink=0.8, aspect=30, pad=0.1, label="Mean squared error")
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Mean squared error", fontsize=8)
    savefig("spatial_error")

    # ── Fig 14: Example means (well column marked, observed sum in title) ──────
    i_show = i_shows[count % len(i_shows)]
    panels = [
        (marginals[i_show, :, :, 1], r"Exact MRF E$[\mathbf{x} \mid d]$"),
        (mean_unweighted[i_show],    r"ViT MC $\hat{\mathrm{E}}[\mathbf{x} \mid d]$"),
        (mean_weighted[i_show],      r"IS-ViT MC $\hat{\mathrm{E}}[\mathbf{x} \mid d]$"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(3, 4))
    for ax, (img, lbl) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap="gray", interpolation="none", origin="upper", vmin=0, vmax=1)
        ax.axvline(obs_col, color="cyan", linewidth=0.8, alpha=0.8)  # the observed well
        ax.set_title(lbl, fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{model}: well col {obs_col}, d = {int(obs_sum[i_show])}", fontsize=8)
    plt.tight_layout()
    savefig("example_means")

print("\nAll figures saved to figures_summed/")
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures_gaussexp", exist_ok=True)

EXPERIMENTS = {
    "A_wells":   ("A", "well observations"),
    "A_random":  ("A", "random observations"),
    "B_wells":   ("B", "well observations"),
    "B_random":  ("B", "random observations"),
    "C_wells":   ("C", "well observations"),
    "C_random":  ("C", "random observations"),
    "D_wells":   ("D", "well observations"),
    "D_random":  ("D", "random observations"),
}

count = 0

for tag, (model, obs_type) in EXPERIMENTS.items():
    title_tag = f"Model {model}, {obs_type}"
    print(f"\n=== {title_tag} ===")
    if obs_type=="well observations":
        title_tag = f"{model}, wells"
    else:
        title_tag = f"{model}, random"

    base = f"data/gauss_exp"
    data         = np.load(f"{base}/obs_{tag}.npz")
    exact        = np.load(f"{base}/exact_{tag}.npz")
    mrf_data     = np.load(f"{base}/mrf_{tag}.npz")
    npz          = np.load(f"{base}/vit_{tag}.npz")

    true_images  = data["images"]
    marginals    = exact["marginals"]
    mrf_imgs     = mrf_data["images"]

    imgs           = npz["samples"]
    log_lik_cond   = npz["log_lik_cond"]
    log_lik_uncond = npz["log_lik_uncond"]
    log_marginal_sum = npz["log_marginal_sum"]
    
    obs_rows = data["obs_rows"]   # (N, K)
    obs_cols = data["obs_cols"]   # (N, K)
    obs_vals = data["obs_vals"]   # (N, K)
    K = obs_rows.shape[1]

    n, N, H, W = imgs.shape
    print(f"  {n} experiments, {N} samples each, {H}x{W}")

    # ── IS weights & ESS ──────────────────────────────────────────────────────
    log_w = log_lik_uncond - log_lik_cond - log_marginal_sum
    log_w -= log_w.max(axis=1, keepdims=True)
    w      = np.exp(log_w)
    w     /= w.sum(axis=1, keepdims=True)
    ess    = 1.0 / (w ** 2).sum(axis=1)

    # ── Posterior means ───────────────────────────────────────────────────────
    mean_unweighted = imgs.mean(axis=1)
    mean_weighted   = (w[:, :, None, None] * imgs).sum(axis=1)
    mean_mrf        = mrf_imgs.mean(axis=1)
    mean_mrf_ess    = np.stack([
        mrf_imgs[i, :int(ess[i])].mean(axis=0) for i in range(n)
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
    n_better      = np.sum(MSEs_weighted < MSEs_unweighted)
    n_worse_mrf   = np.sum(MSEs_weighted >= MSEs_mrf)
    n_worse_mrf_ess = np.sum(MSEs_weighted >= MSEs_mrf_ess)
    print(f"  Mean unweighted MSE         : {MMSE_u:.6f}")
    print(f"  Mean IS-weighted MSE        : {MMSE_w:.6f}")
    print(f"  Mean MRF MSE                : {MMSE_m:.6f}")
    print(f"  Ratio weighted/unweighted   : {MMSE_w/MMSE_u:.4f}")
    print(f"  IS better in                : {100*n_better/n:.1f}% of cases")
    print(f"  IS worse than MRF in        : {100*n_worse_mrf/n:.1f}% of cases")
    print(f"  IS worse than ESS-MRF in    : {100*n_worse_mrf_ess/n:.1f}% of cases")

    def savefig(name):
        plt.savefig(f"figures_gaussexp/{name}_{tag}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── Fig 1: Marginals grid ─────────────────────────────────────────────────
    n_show = min(3, n)
    fig, axes = plt.subplots(n_show, 3, figsize=(7, 1.5 * n_show))
    for j, col_title in enumerate(["Exact marginal", "Unweighted mean", "IS-weighted mean"]):
        axes[0, j].set_title(col_title, fontsize=9)
    for i in range(n_show):
        axes[i, 0].imshow(marginals[i, :, :, 1], cmap="gray", interpolation="none", origin="upper")
        axes[i, 1].imshow(mean_unweighted[i],     cmap="gray", interpolation="none", origin="upper")
        axes[i, 2].imshow(mean_weighted[i],        cmap="gray", interpolation="none", origin="upper")
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
    lo = min(MSEs_unweighted.min(), MSEs_weighted.min()) - 5e-5
    hi_x = MSEs_unweighted.max() + 5e-5
    hi_y = MSEs_weighted.max() + 5e-5
    hi = min(hi_y, hi_x)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, alpha=0.7)
    sc = ax.scatter(MSEs_unweighted, MSEs_weighted, s=12, c=ess, cmap="viridis", alpha=0.7)
    plt.colorbar(sc, ax=ax, label="ESS")
    ax.set_xlim(lo, hi_x); ax.set_ylim(lo, hi_y)
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

    # ── Fig 10: Scatter weighted vs ESS-MRF (colored by ESS) ─────────────────
    lo = min(MSEs_mrf_ess.min(), MSEs_weighted.min()) - 5e-5
    hi_y = MSEs_mrf_ess.max() + 5e-5
    hi_x = MSEs_weighted.max() + 5e-5
    hi = min(hi_y, hi_x)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, alpha=0.7)
    sc = ax.scatter(MSEs_weighted, MSEs_mrf_ess, s=12, c=ess, cmap="viridis", alpha=0.7)
    plt.colorbar(sc, ax=ax, label="ESS")
    ax.set_xlim(lo, hi_x); ax.set_ylim(lo, hi_y)
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
    cbar = plt.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal",
                        shrink=0.8, aspect=30, pad=0.1, label="Mean squared error")
    
    cbar.ax.tick_params(labelsize=8)        # tick numbers
    cbar.set_label("Mean squared error", fontsize=8)  # label text
    savefig("spatial_error")
    
    i_shows = [2, 0, 2, 2, 0, 1, 0, 2]
    i_show = i_shows[count]
    count += 1
    
    obs_display = np.full((H, W), 0.5)
    for j in range(K):
        obs_display[obs_rows[i_show, j], obs_cols[i_show, j]] = obs_vals[i_show, j]
    
    panels = [
        (marginals[i_show,:,:,1], r"Exact MRF E$[\mathbf{x} \mid \mathbf{y}]$"),
        (mean_unweighted[i_show],   r"ViT MC $\hat{\text{E}}[\mathbf{x} \mid \mathbf{y}]$"),
        (mean_weighted[i_show],   r"IS-ViT MC $\hat{\text{E}}[\mathbf{x} \mid \mathbf{y}]$"),
    ]
    
    fig, axes = plt.subplots(3, 1, figsize=(2, 2.67))
    for ax, (img, lbl) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap="gray", interpolation="none", origin="upper", vmin=0, vmax=1)
        ax.set_title(lbl, fontsize=8)
        ax.axis("off")
    
    plt.tight_layout()
    savefig("example_means")

print("\nAll figures saved to figures/")
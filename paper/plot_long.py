#!/usr/bin/env python
# coding: utf-8
"""
Figures for the long-run COMPACT outputs (conditional_model_long_*).

Each vit_long_*.npz stores only per-image reductions:
    mean_unweighted [N,H,W], mean_weighted [N,H,W], ess [N]
so the IS weights are already applied. The exact posterior mean (target) is read
from the matching exact_*.npz; the observation metadata (for display) from obs_*.

Produces only four figure types per experiment, into figures_long/:
    mse_scatter_uw_vs_w   IS-weighted vs unweighted per-image MSE
    ess                   ESS histogram
    spatial_error         per-pixel mean squared error (unweighted vs weighted)
    example_means         exact / unweighted / weighted mean for one example image

Covers both the Gaussian experiment (gauss_exp, wells+random) and the summed
experiment (summed_exp, one well).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures_long", exist_ok=True)

# ── Experiment registry ───────────────────────────────────────────────────────
EXPERIMENTS = []
for M in "ABCD":
    for reg in ["wells", "random"]:
        EXPERIMENTS.append(dict(
            family="gauss", model=M, regime=reg, tag=f"gauss_{M}_{reg}",
            vit=f"data/gauss_exp/vit_long_{M}_{reg}.npz",
            exact=f"data/gauss_exp/exact_{M}_{reg}.npz",
            obs=f"data/gauss_exp/obs_{M}_{reg}.npz",
        ))
for M in "ABCD":
    EXPERIMENTS.append(dict(
        family="summed", model=M, regime="well", tag=f"summed_{M}_well",
        vit=f"data/summed_exp/vit_long_{M}_well.npz",
        exact=f"data/summed_exp/exact_{M}_well.npz",
        obs=f"data/summed_exp/obs_{M}_well.npz",
    ))

# index of the example image shown in the example_means panel (per tag; default 0)
EXAMPLE_IDX = {}

for exp in EXPERIMENTS:
    tag = exp["tag"]
    if not os.path.exists(exp["vit"]):
        print(f"[skip] {tag}: {exp['vit']} not found")
        continue
    print(f"\n=== {tag} ===")

    vit   = np.load(exp["vit"])
    exact = np.load(exp["exact"])
    obs   = np.load(exp["obs"])

    mean_unweighted = vit["mean_unweighted"]   # [N,H,W]
    mean_weighted   = vit["mean_weighted"]     # [N,H,W]
    ess             = vit["ess"]               # [N]
    target          = exact["marginals"][:, :, :, 1]   # [N,H,W]

    N, H, W = mean_unweighted.shape
    print(f"  {N} images, {H}x{W}; mean ESS {ess.mean():.1f}")

    MSEs_unweighted = ((mean_unweighted - target) ** 2).mean(axis=(1, 2))
    MSEs_weighted   = ((mean_weighted   - target) ** 2).mean(axis=(1, 2))
    n_better      = np.sum(MSEs_weighted < MSEs_unweighted)
    n_total = len(MSEs_weighted)
    percent_better = 100*n_better/n_total
    print(f"  Mean unweighted MSE : {MSEs_unweighted.mean():.6f}")
    print(f"  Mean IS-weighted MSE: {MSEs_weighted.mean():.6f}")
    print(f"  Ratio w/uw          : {MSEs_weighted.mean()/MSEs_unweighted.mean():.4f}")
    print(f"  Improvement         : {1 - MSEs_weighted.mean()/MSEs_unweighted.mean():.4f}")
    print(f"  IS-weighted better in {percent_better}% of cases")

    # summed needs the well column for display
    obs_col = int(obs["obs_col"]) if "obs_col" in obs.files else None
    obs_sum = obs["obs_sum"] if "obs_sum" in obs.files else None

    def savefig(name):
        plt.savefig(f"figures_long/{name}_{tag}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── Fig: scatter unweighted vs weighted ───────────────────────────────────
    lo = min(MSEs_weighted.min(), MSEs_unweighted.min()) - 5e-5
    hi = max(MSEs_weighted.max(), MSEs_unweighted.max()) + 5e-5
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, alpha=0.7)
    sc = ax.scatter(MSEs_unweighted, MSEs_weighted, s=12, c=ess, cmap="viridis", alpha=0.8)
    plt.colorbar(sc, ax=ax, label="ESS")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Unweighted MSE"); ax.set_ylabel("IS-weighted MSE")
    ax.grid(True, linewidth=0.4)
    plt.tight_layout()
    savefig("mse_scatter_uw_vs_w")
    
    if exp["family"]=="summed":
        d = obs["obs_sum"]
        d_clust = np.minimum(d, 12 - d)
        
        lo = min(MSEs_weighted.min(), MSEs_unweighted.min()) - 5e-5
        hi = max(MSEs_weighted.max(), MSEs_unweighted.max()) + 5e-5
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, alpha=0.7)
        sc = ax.scatter(MSEs_unweighted, MSEs_weighted, s=12, c=d_clust, cmap="viridis", alpha=0.8)
        plt.colorbar(sc, ax=ax, label="min(d, 12-d)")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Unweighted MSE"); ax.set_ylabel("IS-weighted MSE")
        ax.grid(True, linewidth=0.4)
        plt.tight_layout()
        savefig("mse_scatter_uw_vs_w_d")
        
        # ── Fig: spatial error by d_clust ─────────────────────────────────────
        d_vals = [dv for dv in sorted(np.unique(d_clust)) if dv in {0, 3, 6}]
        n_d = len(d_vals)
        fig, axes = plt.subplots(n_d, 2, figsize=(3, 1.5 * n_d), constrained_layout=True)
        if n_d == 1:
            axes = axes[np.newaxis, :]
        vmax = 0.0
        grids = []
        for dv in d_vals:
            mask = d_clust == dv
            eu = ((mean_unweighted[mask] - target[mask]) ** 2).mean(axis=0)
            ew = ((mean_weighted[mask]   - target[mask]) ** 2).mean(axis=0)
            grids.append((dv, eu, ew))
            vmax = max(vmax, eu.max(), ew.max())
        for row, (dv, eu, ew) in enumerate(grids):
            for col, (err, lbl) in enumerate([(eu, "Unweighted"), (ew, "IS-weighted")]):
                ax = axes[row, col]
                im = ax.imshow(err, cmap="hot", interpolation="none",
                               origin="upper", vmin=0, vmax=vmax)
                ax.axis("off")
                if obs_col is not None:
                    ax.axvline(obs_col, color="cyan", linewidth=0.6, alpha=0.7)
                if row == 0:
                    ax.set_title(lbl, fontsize=9)
            axes[row, 0].text(-0.05, 0.5, f"min(d,12-d)={int(dv)}", transform=axes[row, 0].transAxes,
                  fontsize=8, va="center", ha="right", rotation=90)
        cbar = plt.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal",
                            shrink=0.8, aspect=30, pad=0.05)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Mean squared error", fontsize=8)
        savefig("spatial_error_by_d")
        
        d_vals = sorted(np.unique(d))
        mse_uw_d, mse_w_d, d_colors, d_labels = [], [], [], []
        for dv in d_vals:
            mask = d == dv
            t  = target[mask].mean(axis=0)
            uw = mean_unweighted[mask].mean(axis=0)
            w  = mean_weighted[mask].mean(axis=0)
            mse_uw_d.append(((uw - t) ** 2).mean())
            mse_w_d.append(((w  - t) ** 2).mean())
            d_labels.append(dv)
            d_colors.append(np.minimum(dv, 12 - dv))
    
        mse_uw_d = np.array(mse_uw_d)
        mse_w_d  = np.array(mse_w_d)
        d_labels = np.array(d_labels)
        d_colors = np.array(d_colors)
    
        lo = min(mse_uw_d.min(), mse_w_d.min()) - 5e-6
        hi = max(mse_uw_d.max(), mse_w_d.max()) + 5e-6
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, alpha=0.7)
        sc = ax.scatter(mse_uw_d, mse_w_d, s=40, c=d_colors, cmap="viridis", zorder=3)
        for i, dv in enumerate(d_labels):
            ax.annotate(str(int(dv)), (mse_uw_d[i], mse_w_d[i]),
                        textcoords="offset points", xytext=(5, 3), fontsize=7)
        plt.colorbar(sc, ax=ax, label="min(d, 12-d)")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Unweighted MSE"); ax.set_ylabel("IS-weighted MSE")
        ax.grid(True, linewidth=0.4)
        plt.tight_layout()
        savefig("mse_scatter_per_d")
        

    # ── Fig: ESS histogram ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(3.5, 3))
    ax.hist(ess, bins=20, density=True, color="darkorange", edgecolor="white")
    ax.axvline(ess.mean(), color="black", linewidth=0.8, linestyle="--",
               label=f"mean {ess.mean():.0f}")
    ax.set_xlabel("ESS"); ax.set_ylabel("Density")
    ax.legend(fontsize=8); ax.grid(True, linewidth=0.4)
    plt.tight_layout()
    savefig("ess")

    # ── Fig: spatial error maps ────────────────────────────────────────────────
    err_unweighted = ((mean_unweighted - target) ** 2).mean(axis=0)
    err_weighted   = ((mean_weighted   - target) ** 2).mean(axis=0)
    vmax = max(err_unweighted.max(), err_weighted.max())
    fig, axes = plt.subplots(1, 2, figsize=(3, 1.5), constrained_layout=True)
    for ax, err, lbl in zip(axes, [err_unweighted, err_weighted],
                            ["Unweighted", "IS-weighted"]):
        im = ax.imshow(err, cmap="hot", interpolation="none",
                       origin="upper", vmin=0, vmax=vmax)
        ax.set_title(lbl, fontsize=9); ax.axis("off")
        if obs_col is not None:
            ax.axvline(obs_col, color="cyan", linewidth=0.6, alpha=0.7)
    cbar = plt.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal",
                        shrink=0.8, aspect=30, pad=0.1)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Mean squared error", fontsize=8)
    savefig("spatial_error")

    # ── Fig: example means ─────────────────────────────────────────────────────
    i_show = EXAMPLE_IDX.get(tag, 0)
    cond = "d" if exp["family"] == "summed" else r"\mathbf{y}"
    panels = [
        (target[i_show],          rf"Exact MRF E$[\mathbf{{x}} \mid {cond}]$"),
        (mean_unweighted[i_show], rf"ViT MC $\hat{{\mathrm{{E}}}}[\mathbf{{x}} \mid {cond}]$"),
        (mean_weighted[i_show],   rf"IS-ViT MC $\hat{{\mathrm{{E}}}}[\mathbf{{x}} \mid {cond}]$"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(3, 4))
    for ax, (img, lbl) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap="gray", interpolation="none", origin="upper", vmin=0, vmax=1)
        if obs_col is not None:
            ax.axvline(obs_col, color="cyan", linewidth=0.8, alpha=0.8)
        ax.set_title(lbl, fontsize=8)
        ax.axis("off")
    if exp["family"] == "summed" and obs_sum is not None:
        fig.suptitle(f"{exp['model']}: well col {obs_col}, d = {int(obs_sum[i_show])}",
                     fontsize=8)
    else:
        fig.suptitle(f"{exp['model']}, {exp['regime']}", fontsize=8)
    plt.tight_layout()
    savefig("example_means")

print("\nAll figures saved to figures_long/")
# -*- coding: utf-8 -*-
"""
Master figure script for Chapter 4.

Produces one image per model for every figure type, so that the LaTeX
chapter can assemble them into subfigure environments where panel (a)
is always Model I, panel (b) Model II.

Models B and C have been dropped for the paper; former Model A is now
Model I and former Model D is now Model II. Data files on disk keep
their original A/D naming (loaded via MODELS below), but all saved
figures, titles, and printed summaries use the paper's I/II labels
(see OUTPUT_LABEL).

Output layout (all files written to ./figures/):

  samples_model_{X}.png                – MRF vs ViT sample grid
  cluster_overlay_model_{X}.png        – same grid with connected components coloured
  pixel_proportion_model_{X}.png       – QQ + histogram of pixel proportion
  cluster_histogram_model_{X}.png      – cluster-count histogram
  loglik_under_mrf_model_{X}.png       – log-likelihood histograms under MRF evaluator
  loglik_under_vit_model_{X}.png       – log-likelihood histograms under ViT evaluator
  scatter_mrf_model_{X}.png            – log p_ViT vs log p_MRF, MRF samples
  scatter_vit_model_{X}.png            – log p_MRF vs log p_ViT, ViT samples
  ratio_mrf_model_{X}.png              – density-ratio histogram, MRF samples
  ratio_vit_model_{X}.png              – density-ratio histogram, ViT samples
  training_loss_model_{X}.png          – training loss curve

where X ∈ {I, II}.

Colour convention: 0 → black, 1 → white  (cmap="gray").
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from scipy.ndimage import label
from scipy import stats

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR    = os.path.join(SCRIPT_DIR, "figures_paper")
os.makedirs(FIG_DIR, exist_ok=True)

MODELS = ["A", "D"]

# Data files on disk still use the original A/D naming; this maps each to
# the paper's renamed label used for output filenames, titles, and print-outs.
OUTPUT_LABEL = {"A": "I", "D": "II"}

# ── style ───────────────────────────────────────────────────────────────────
MRF_COLOR = "#E07B39"
VIT_COLOR  = "#4C9BE8"
ALPHA      = 0.65
DPI        = 200

plt.rcParams.update({
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "legend.fontsize":  11,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "figure.dpi":       DPI,
})

FS_SINGLE  = (4.5, 3.6)
FS_SCATTER = (4.2, 4.0)
FS_TWOPAN  = (8.0, 3.6)

SCATTER_PT = 5

MODEL_TITLES = {
    "A": "Model I \u2013 stationary Ising",
    "D": "Model II \u2013 anisotropic 2nd-order",
}

# ── loaders ─────────────────────────────────────────────────────────────────

def load_images(model):
    mrf = np.load(os.path.join(SCRIPT_DIR, f"data/mrf-model-{model}.npz"))["images"]
    vit = np.load(os.path.join(SCRIPT_DIR, f"data/ViT-model-{model}.npz"))["images"]
    return mrf, vit


def load_logliks(model):
    return np.load(os.path.join(SCRIPT_DIR, f"logliks-model-{model}.npz"))


def load_loss(model):
    path = os.path.join(SCRIPT_DIR, f"outputs-model-{model}/epoch_loss.csv")
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        return np.arange(1, len(data) + 1), data
    return data[:, 0], data[:, 1]


# ── helpers ──────────────────────────────────────────────────────────────────

def fig_path(name):
    return os.path.join(FIG_DIR, name)


def nn_agreement(images):
    agree_h = (images[:, :, :-1] == images[:, :, 1:]).mean()
    agree_v = (images[:, :-1, :] == images[:, 1:, :]).mean()
    return 0.5 * (agree_h + agree_v)


def percolates(img, direction):
    lbl, _ = label(img)
    if direction == "short":
        a = set(lbl[0,  :].flatten()) - {0}
        b = set(lbl[-1, :].flatten()) - {0}
    else:
        a = set(lbl[:, 0].flatten())  - {0}
        b = set(lbl[:, -1].flatten()) - {0}
    return bool(a & b)


def percolation_prob(images, direction):
    return float(np.mean([percolates(img, direction) for img in images]))


def count_clusters(images):
    return np.array([label(img)[1] for img in images])


def regress_one(x, y):
    slope, intercept, r, _, se = stats.linregress(x, y)
    c_hat      = float(np.mean(y - x))
    resid_unit = y - (x + c_hat)
    sd_unit    = float(np.std(resid_unit, ddof=1))
    return {
        "slope": float(slope), "slope_se": float(se),
        "intercept": float(intercept), "r2": float(r ** 2),
        "c_hat": c_hat, "sd_unit": sd_unit,
        "resid_unit": resid_unit,
    }


def make_cluster_rgb(img):
    """Return an RGB array where each foreground connected component is
    assigned a distinct hue and the background (pixel==0) is white."""
    labeled, n_clusters = label(img)
    # Build a colormap with n_clusters distinct hues (skip very light colours).
    # Cycle through HSV hues spaced evenly; use saturation/value that stay
    # clearly visible against a white background.
    rgb = np.ones((*labeled.shape, 3), dtype=float)   # white background
    if n_clusters == 0:
        return rgb
    hues = np.linspace(0, 1, n_clusters + 1)[:-1]
    for k, h in enumerate(hues, start=1):
        # HSV → RGB; saturation=0.75, value=0.82
        r, g, b = mcolors.hsv_to_rgb([h, 0.75, 0.82])
        mask = labeled == k
        rgb[mask, 0] = r
        rgb[mask, 1] = g
        rgb[mask, 2] = b
    return rgb


# ════════════════════════════════════════════════════════════════════════════
# Figure: sample grid  (MRF left | ViT right)
#
# Layout: two rows of sample panels, followed by one row of cluster-coloured
# panels, for each half (MRF / ViT).  A labelled divider separates the two
# halves.
# ════════════════════════════════════════════════════════════════════════════

def plot_sample_grid(model):
    label = OUTPUT_LABEL[model]
    mrf, vit = load_images(model)
    ncols, nrows = 2, 3          # 2 columns × 3 rows per half = 6 images each
    H, W = mrf.shape[1], mrf.shape[2]
    aspect = W / H               # = 2 for 12×24 images

    wspace = 0.05
    hspace = wspace * aspect

    div_w = aspect * 0.30
    col_widths = [aspect] * ncols + [div_w] + [aspect] * ncols

    img_h_in = 1.10
    img_w_in = img_h_in * aspect
    fig_w = 2 * ncols * img_w_in + img_h_in * div_w + 0.15
    fig_h = nrows * img_h_in + 0.80

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    top_margin = 0.72 / fig_h
    gs = gridspec.GridSpec(
        nrows=nrows, ncols=2 * ncols + 1,
        figure=fig,
        width_ratios=col_widths,
        wspace=wspace,
        hspace=hspace,
        top=1.0 - top_margin,
        bottom=0.02,
        left=0.01,
        right=0.99,
    )

    for r in range(nrows):
        for c in range(ncols):
            idx = r * ncols + c
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(mrf[idx], cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.axis("off")
            ax = fig.add_subplot(gs[r, c + ncols + 1])
            ax.imshow(vit[idx], cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.axis("off")

    for r in range(nrows):
        ax = fig.add_subplot(gs[r, ncols])
        ax.set_visible(False)

    fig.canvas.draw()

    ax_line = fig.add_axes([0, 0, 1, 1], facecolor="none")
    ax_line.set_xlim(0, 1); ax_line.set_ylim(0, 1)
    ax_line.axis("off")

    ss_last_mrf  = gs[0, ncols - 1].get_position(fig)
    ss_first_vit = gs[0, ncols + 1].get_position(fig)
    div_x = 0.5 * (ss_last_mrf.x1 + ss_first_vit.x0)
    ax_line.axvline(div_x, color="0.55", lw=0.8, ls="--",
                    ymin=0.02, ymax=0.80)

    top_y    = gs[0, 0].get_position(fig).y1
    ss_mrf_l = gs[0, 0].get_position(fig)
    ss_mrf_r = gs[0, ncols - 1].get_position(fig)
    ss_vit_l = gs[0, ncols + 1].get_position(fig)
    ss_vit_r = gs[0, 2 * ncols].get_position(fig)

    half_label_y = top_y + 0.01
    ax_line.text(0.5 * (ss_mrf_l.x0 + ss_mrf_r.x1), half_label_y,
                 "MRF samples", ha="center", va="bottom",
                 fontsize=13, color="0.35", transform=fig.transFigure)
    ax_line.text(0.5 * (ss_vit_l.x0 + ss_vit_r.x1), half_label_y,
                 "ViT samples", ha="center", va="bottom",
                 fontsize=13, color="0.35", transform=fig.transFigure)

    plt.savefig(fig_path(f"samples_model_{label}.png"),
                bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  samples_model_{label}.png")


# ════════════════════════════════════════════════════════════════════════════
# Figure: cluster overlay grid  (same layout as sample grid, colours by
#         connected component via scipy.ndimage.label)
# ════════════════════════════════════════════════════════════════════════════

def plot_cluster_overlay(model):
    label = OUTPUT_LABEL[model]
    mrf, vit = load_images(model)
    ncols, nrows = 2, 3
    H, W = mrf.shape[1], mrf.shape[2]
    aspect = W / H

    wspace = 0.05
    hspace = wspace * aspect

    div_w = aspect * 0.30
    col_widths = [aspect] * ncols + [div_w] + [aspect] * ncols

    img_h_in = 1.10
    img_w_in = img_h_in * aspect
    fig_w = 2 * ncols * img_w_in + img_h_in * div_w + 0.15
    fig_h = nrows * img_h_in + 0.80

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    top_margin = 0.72 / fig_h
    gs = gridspec.GridSpec(
        nrows=nrows, ncols=2 * ncols + 1,
        figure=fig,
        width_ratios=col_widths,
        wspace=wspace,
        hspace=hspace,
        top=1.0 - top_margin,
        bottom=0.02,
        left=0.01,
        right=0.99,
    )

    for r in range(nrows):
        for c in range(ncols):
            idx = r * ncols + c
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(make_cluster_rgb(mrf[idx]), interpolation="nearest")
            ax.axis("off")
            ax = fig.add_subplot(gs[r, c + ncols + 1])
            ax.imshow(make_cluster_rgb(vit[idx]), interpolation="nearest")
            ax.axis("off")

    for r in range(nrows):
        ax = fig.add_subplot(gs[r, ncols])
        ax.set_visible(False)

    fig.canvas.draw()

    ax_line = fig.add_axes([0, 0, 1, 1], facecolor="none")
    ax_line.set_xlim(0, 1); ax_line.set_ylim(0, 1)
    ax_line.axis("off")

    ss_last_mrf  = gs[0, ncols - 1].get_position(fig)
    ss_first_vit = gs[0, ncols + 1].get_position(fig)
    div_x = 0.5 * (ss_last_mrf.x1 + ss_first_vit.x0)
    ax_line.axvline(div_x, color="0.55", lw=0.8, ls="--",
                    ymin=0.02, ymax=0.80)

    top_y    = gs[0, 0].get_position(fig).y1
    ss_mrf_l = gs[0, 0].get_position(fig)
    ss_mrf_r = gs[0, ncols - 1].get_position(fig)
    ss_vit_l = gs[0, ncols + 1].get_position(fig)
    ss_vit_r = gs[0, 2 * ncols].get_position(fig)

    half_label_y = top_y + 0.01
    ax_line.text(0.5 * (ss_mrf_l.x0 + ss_mrf_r.x1), half_label_y,
                 "MRF samples", ha="center", va="bottom",
                 fontsize=13, color="0.35", transform=fig.transFigure)
    ax_line.text(0.5 * (ss_vit_l.x0 + ss_vit_r.x1), half_label_y,
                 "ViT samples", ha="center", va="bottom",
                 fontsize=13, color="0.35", transform=fig.transFigure)

    fig.text(0.5, 0.99, MODEL_TITLES[model],
             ha="center", va="top", fontsize=14)

    plt.savefig(fig_path(f"cluster_overlay_model_{label}.png"),
                bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  cluster_overlay_model_{label}.png")


# ════════════════════════════════════════════════════════════════════════════
# Figure: pixel proportion  (QQ + histogram, two-panel)
# ════════════════════════════════════════════════════════════════════════════

def plot_pixel_proportion(model):
    label = OUTPUT_LABEL[model]
    mrf, vit = load_images(model)
    prop_m = mrf.mean(axis=(1, 2))
    prop_v = vit.mean(axis=(1, 2))

    fig, (ax_qq, ax_h) = plt.subplots(1, 2, figsize=FS_TWOPAN)

    q_m = np.quantile(prop_m, np.linspace(0, 1, 200))
    q_v = np.quantile(prop_v, np.linspace(0, 1, 200))
    lim = (min(q_m.min(), q_v.min()) - 0.02, max(q_m.max(), q_v.max()) + 0.02)
    ax_qq.plot(lim, lim, "k--", lw=1.2, label="$y = x$")
    ax_qq.scatter(q_m, q_v, s=12, alpha=0.7, color=VIT_COLOR)
    ax_qq.set_xlim(lim); ax_qq.set_ylim(lim)
    ax_qq.set_xlabel("MRF quantiles"); ax_qq.set_ylabel("ViT quantiles")
    ax_qq.set_title("QQ plot"); ax_qq.legend(); ax_qq.grid(True, lw=0.4)

    bins = np.linspace(0, 1, 30)
    ax_h.hist(prop_m, bins=bins, alpha=ALPHA, color=MRF_COLOR, label="MRF", density=True)
    ax_h.hist(prop_v, bins=bins, alpha=ALPHA, color=VIT_COLOR,  label="ViT",  density=True)
    ax_h.set_xlabel("Pixel proportion"); ax_h.set_ylabel("Density")
    ax_h.set_title("Histogram"); ax_h.legend(); ax_h.grid(True, lw=0.4)

    fig.suptitle(MODEL_TITLES[model])
    plt.tight_layout()
    plt.savefig(fig_path(f"pixel_proportion_model_{label}.png"), bbox_inches="tight")
    plt.close()
    print(f"  pixel_proportion_model_{label}.png")


# ════════════════════════════════════════════════════════════════════════════
# Figure: cluster-count histogram
# ════════════════════════════════════════════════════════════════════════════

def plot_cluster_histogram(model):
    label = OUTPUT_LABEL[model]
    mrf, vit = load_images(model)
    c_m = count_clusters(mrf)
    c_v = count_clusters(vit)

    bins = np.arange(0, max(c_m.max(), c_v.max()) + 3) - 0.5

    fig, ax = plt.subplots(figsize=FS_SINGLE)
    ax.hist(c_m, bins=bins, alpha=ALPHA, color=MRF_COLOR, label="MRF", density=True)
    ax.hist(c_v, bins=bins, alpha=ALPHA, color=VIT_COLOR,  label="ViT",  density=True)
    ax.set_xlabel("Number of clusters"); ax.set_ylabel("Density")
    ax.set_title(MODEL_TITLES[model]); ax.legend(); ax.grid(True, lw=0.4)
    plt.tight_layout()
    plt.savefig(fig_path(f"cluster_histogram_model_{label}.png"), bbox_inches="tight")
    plt.close()
    print(f"  cluster_histogram_model_{label}.png")


# ════════════════════════════════════════════════════════════════════════════
# Figure: log-likelihood histograms under each evaluator
# ════════════════════════════════════════════════════════════════════════════

def plot_loglik_histograms(model):
    label = OUTPUT_LABEL[model]
    d = load_logliks(model)
    for evaluator, key_m, key_v in [
        ("mrf", "mrf_under_exact", "vit_under_exact"),
        ("vit", "mrf_under_vit",   "vit_under_vit"),
    ]:
        ll_m = d[key_m]; ll_v = d[key_v]
        all_vals = np.concatenate([ll_m, ll_v])
        bins = np.linspace(all_vals.min(), all_vals.max(), 40)

        fig, ax = plt.subplots(figsize=FS_SINGLE)
        ax.hist(ll_m, bins=bins, alpha=ALPHA, color=MRF_COLOR, label="MRF samples")
        ax.hist(ll_v, bins=bins, alpha=ALPHA, color=VIT_COLOR,  label="ViT samples")
        ax.set_xlabel("Log-likelihood"); ax.set_ylabel("Count")
        ax.set_title(MODEL_TITLES[model]); ax.legend(); ax.grid(True, lw=0.4)
        plt.tight_layout()
        fname = f"loglik_under_{evaluator}_model_{label}.png"
        plt.savefig(fig_path(fname), bbox_inches="tight")
        plt.close()
        print(f"  {fname}")


# ════════════════════════════════════════════════════════════════════════════
# Figure: log-likelihood scatter  (one per source)
# ════════════════════════════════════════════════════════════════════════════

def plot_loglik_scatter(model):
    label = OUTPUT_LABEL[model]
    d = load_logliks(model)

    configs = [
        (d["mrf_under_exact"], d["mrf_under_vit"],
         r"$\log p_{\mathrm{MRF}}(x)$", r"$\log p_{\mathrm{ViT}}(x)$",
         MRF_COLOR, f"scatter_mrf_model_{label}.png"),
        (d["vit_under_vit"],   d["vit_under_exact"],
         r"$\log p_{\mathrm{ViT}}(x)$", r"$\log p_{\mathrm{MRF}}(x)$",
         VIT_COLOR, f"scatter_vit_model_{label}.png"),
    ]

    for x, y, xl, yl, color, fname in configs:
        reg = regress_one(x, y)

        lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
        pad = 0.02 * (hi - lo)
        xs = np.array([lo - pad, hi + pad])

        fig, ax = plt.subplots(figsize=FS_SCATTER)
        ax.scatter(x, y, s=SCATTER_PT, alpha=0.4, color=color, rasterized=True)
        ax.plot(xs, xs, "k--", lw=1.2,
                label="slope 1")
        ax.plot(xs, reg["slope"] * xs + reg["intercept"], "r-", lw=1.4,
                label=f"slope {reg['slope']:.3f},  $R^2={reg['r2']:.4f}$")
        ax.set_xlim(xs); ax.set_ylim(xs)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.legend(loc="upper left"); ax.grid(True, lw=0.4)
        plt.tight_layout()
        plt.savefig(fig_path(fname), bbox_inches="tight")
        plt.close()
        print(f"  {fname}")


# ════════════════════════════════════════════════════════════════════════════
# Figure: density-ratio histograms
# ════════════════════════════════════════════════════════════════════════════

def plot_ratio_histograms(model):
    label = OUTPUT_LABEL[model]
    d = load_logliks(model)

    configs = [
        (d["mrf_under_vit"]   - d["mrf_under_exact"],
         r"$p_{\mathrm{ViT}}(x)/p_{\mathrm{MRF}}(x)$, MRF samples",
         MRF_COLOR, f"ratio_mrf_model_{label}.png"),
        (d["vit_under_exact"] - d["vit_under_vit"],
         r"$p_{\mathrm{MRF}}(x)/p_{\mathrm{ViT}}(x)$, ViT samples",
         VIT_COLOR, f"ratio_vit_model_{label}.png"),
    ]

    for log_ratio, title_frag, color, fname in configs:
        ratios = np.exp(log_ratio)
        hist, bins = np.histogram(ratios, bins=30)
        logbins = np.logspace(
            np.log10(max(bins[0], 1e-6)),
            np.log10(bins[-1]), len(bins)
        )
        pct = 100 * np.mean(ratios > 1)

        fig, ax = plt.subplots(figsize=FS_SINGLE)
        ax.hist(ratios, bins=logbins, color=color, alpha=0.85)
        ax.axvline(1, color="black", lw=1.5)
        ax.set_xscale("log")
        ax.set_xlabel("Ratio (log scale)"); ax.set_ylabel("Count")
        ax.set_title(f"{MODEL_TITLES[model]}\n{title_frag}  ({pct:.1f}% $> 1$)")
        ax.grid(True, lw=0.4)
        plt.tight_layout()
        plt.savefig(fig_path(fname), bbox_inches="tight")
        plt.close()
        print(f"  {fname}")


# ════════════════════════════════════════════════════════════════════════════
# Figure: training loss
# ════════════════════════════════════════════════════════════════════════════

def plot_training_loss(model):
    label = OUTPUT_LABEL[model]
    epochs, losses = load_loss(model)
    unif = np.log(16)

    fig, ax = plt.subplots(figsize=FS_SINGLE)
    ax.plot(epochs, losses, color=VIT_COLOR, lw=1.0, label="Training loss")
    ax.axhline(unif, color=MRF_COLOR, ls="--", lw=1.5, label=r"Uniform $\log 16$")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title(MODEL_TITLES[model]); ax.legend(); ax.grid(True, lw=0.4)
    plt.tight_layout()
    plt.savefig(fig_path(f"training_loss_model_{label}.png"), bbox_inches="tight")
    plt.close()
    print(f"  training_loss_model_{label}.png")

def importance_diagnostics(d):
    lw = d["vit_under_exact"] - d["vit_under_vit"]   # log p_MRF - log p_ViT on ViT samples
    w = np.exp(lw - lw.max())                        # rescaled for stability
    n = len(w)
    ess = w.sum() ** 2 / (w ** 2).sum()
    return {
        "n": n,
        "ess": float(ess),
        "ess_frac": float(ess / n),
        "log_wmax": float(lw.max()),
        "max_share": float(w.max() / w.sum()),       # max normalized weight
        "mean_w": float(np.mean(np.exp(lw))),        # ~1 if both densities normalized
    }

# ════════════════════════════════════════════════════════════════════════════
# Summary statistics (printed only)
# ════════════════════════════════════════════════════════════════════════════

def print_summary_stats():
    print("\n" + "=" * 100)
    print(f"{'Model':<7} "
          f"{'NN MRF':>8} {'NN ViT':>8} "
          f"{'Short MRF':>11} {'Short ViT':>11} {'rel %':>8} "
          f"{'Long MRF':>10} {'Long ViT':>10} {'rel %':>8}")
    print("=" * 100)
    for m in MODELS:
        mrf, vit = load_images(m)
        nn_m  = nn_agreement(mrf);              nn_v  = nn_agreement(vit)
        ps_m  = percolation_prob(mrf, "short"); ps_v  = percolation_prob(vit, "short")
        pl_m  = percolation_prob(mrf, "long");  pl_v  = percolation_prob(vit, "long")
        rs    = 100 * (ps_v - ps_m) / ps_m if ps_m > 0 else float("nan")
        rl    = 100 * (pl_v - pl_m) / pl_m if pl_m > 0 else float("nan")
        print(f"  {OUTPUT_LABEL[m]:<5} {nn_m:>8.3f} {nn_v:>8.3f} "
              f"{ps_m:>11.3f} {ps_v:>11.3f} {rs:>+8.1f} "
              f"{pl_m:>10.3f} {pl_v:>10.3f} {rl:>+8.1f}")
    print("=" * 100)

    print()
    print(f"{'Model':<7} {'Source':<12} "
          f"{'slope b':>9} {'SE(b)':>8} {'intercept a':>13} "
          f"{'R^2':>7} {'sd unit':>9} {'sd fit':>9}")
    print("-" * 80)
    for m in MODELS:
        d = load_logliks(m)
        for label_str, x, y in [
            ("MRF samples", d["mrf_under_exact"], d["mrf_under_vit"]),
            ("ViT samples", d["vit_under_vit"],   d["vit_under_exact"]),
        ]:
            reg = regress_one(x, y)
            resid_fit = y - (reg["slope"] * x + reg["intercept"])
            sd_fit = float(np.std(resid_fit, ddof=2))
            print(f"  {OUTPUT_LABEL[m]:<5} {label_str:<12} "
                  f"{reg['slope']:>9.4f} {reg['slope_se']:>8.4f} "
                  f"{reg['intercept']:>13.3f} "
                  f"{reg['r2']:>7.4f} {reg['sd_unit']:>9.3f} {sd_fit:>9.3f}")
        print()
        print(f"{'Model':<7} {'n':>7} {'ESS':>10} {'ESS/n':>8} "
          f"{'log wmax':>10} {'max share':>10} {'mean w':>8}")
    print("-" * 66)
    for m in MODELS:
        s = importance_diagnostics(load_logliks(m))
        print(f"  {OUTPUT_LABEL[m]:<5} {s['n']:>7d} {s['ess']:>10.1f} {s['ess_frac']:>8.3f} "
              f"{s['log_wmax']:>10.2f} {s['max_share']:>10.4f} {s['mean_w']:>8.3f}")

# ════════════════════════════════════════════════════════════════════════════
# Figure: combined log-likelihood scatter  (MRF + ViT samples, one panel)
# ════════════════════════════════════════════════════════════════════════════

def plot_loglik_scatter_combined(model):
    label = OUTPUT_LABEL[model]
    d = load_logliks(model)

    x_m, y_m = d["mrf_under_exact"], d["mrf_under_vit"]   # MRF samples
    x_v, y_v = d["vit_under_exact"], d["vit_under_vit"]   # ViT samples

    all_x = np.concatenate([x_m, x_v])
    all_y = np.concatenate([y_m, y_v])
    lo = min(all_x.min(), all_y.min()); hi = max(all_x.max(), all_y.max())
    pad = 0.02 * (hi - lo)
    xs = np.array([lo - pad, hi + pad])

    fig, ax = plt.subplots(figsize=FS_SCATTER)
    ax.scatter(x_m, y_m, s=SCATTER_PT, alpha=0.4, color=MRF_COLOR,
               label="MRF samples", rasterized=True)
    ax.scatter(x_v, y_v, s=SCATTER_PT, alpha=0.4, color=VIT_COLOR,
               label="ViT samples", rasterized=True)
    ax.plot(xs, xs, "k--", lw=1.2, label="$y = x$")
    ax.set_xlim(xs); ax.set_ylim(xs)
    ax.set_xlabel(r"$\log p_{\mathrm{MRF}}(x)$")
    ax.set_ylabel(r"$\log p_{\mathrm{ViT}}(x)$")
    ax.set_title(MODEL_TITLES[model])
    ax.legend(loc="upper left"); ax.grid(True, lw=0.4)
    plt.tight_layout()
    plt.savefig(fig_path(f"scatter_combined_model_{label}.png"), bbox_inches="tight")
    plt.close()
    print(f"  scatter_combined_model_{label}.png")

from scipy.stats import skew, kurtosis, shapiro

def bridge_logc(lwA, lwB, iters=500, tol=1e-12):
    """Meng-Wong iterative bridge. lwA = log p~/p_theta on MRF samples,
    lwB = log p~/p_theta on ViT samples. Returns log c, c = sum_x p~(x)."""
    nA, nB = len(lwA), len(lwB)
    sA, sB = nA / (nA + nB), nB / (nA + nB)
    lA, lB = np.exp(lwA), np.exp(lwB)
    r = np.mean(lB)
    for _ in range(iters):
        r_new = np.mean(lB / (sA * lB + sB * r)) / np.mean(1.0 / (sA * lA + sB * r))
        done = abs(np.log(r_new / r)) < tol
        r = r_new
        if done:
            break
    return np.log(r)


def normalization_checks(d, n_boot=2000, n_sim=20000, seed=0):
    rng = np.random.default_rng(seed)
    lwB = d["vit_under_exact"] - d["vit_under_vit"]      # ViT samples, log p~/p_theta
    lwA = d["mrf_under_exact"] - d["mrf_under_vit"]      # MRF samples, log p~/p_theta
    n = len(lwB)
    out = {}

    # 1. forward and reverse mean-weight estimates of c, bootstrap CIs
    wB, wA_rev = np.exp(lwB), np.exp(-lwA)               # w on q-samples; p_theta/p~ on p-samples
    bf = np.array([rng.choice(wB, n).mean() for _ in range(n_boot)])
    br = np.array([rng.choice(wA_rev, len(wA_rev)).mean() for _ in range(n_boot)])
    out["mean_w_fwd"] = (wB.mean(), *np.percentile(bf, [2.5, 97.5]))
    out["mean_w_rev"] = (wA_rev.mean(), *np.percentile(br, [2.5, 97.5]))

    # 2. parametric null: lognormal weights with E[w]=1, same sigma, same n
    s = lwB.std(ddof=1)
    sim = np.exp(rng.normal(-s**2 / 2, s, size=(n_sim, n))).mean(axis=1)
    out["p_fwd_lognormal_null"] = float(np.mean(sim <= wB.mean()))
    out["null_mean_w_5_95"] = tuple(np.percentile(sim, [5, 95]))
    out["null_skew_of_sample_mean"] = float(skew(sim))

    # 3. shape of log w (lognormal => Gaussian, zero skew, zero excess kurtosis)
    for name, lw in [("B", lwB), ("A", lwA)]:
        out[f"lw_{name}_mean_sd"] = (lw.mean(), lw.std(ddof=1))
        out[f"lw_{name}_skew_exkurt"] = (skew(lw), kurtosis(lw))
        out[f"lw_{name}_shapiro_p"] = shapiro(lw)[1]

    # 4. Gaussian-based estimates of log c (valid only if lw ~ Gaussian)
    vB, vA = lwB.var(ddof=1), lwA.var(ddof=1)
    logc_gB = lwB.mean() + vB / 2
    logc_gA = lwA.mean() - vA / 2
    se_gB = np.sqrt(vB / n + vB**2 / (2 * (n - 1)))
    se_gA = np.sqrt(vA / len(lwA) + vA**2 / (2 * (len(lwA) - 1)))
    out["logc_gauss_B"] = (logc_gB, se_gB)
    out["logc_gauss_A"] = (logc_gA, se_gA)

    # 5. bridge estimate of log c with bootstrap SE (resample both sets)
    lc = bridge_logc(lwA, lwB)
    lcb = np.array([bridge_logc(rng.choice(lwA, len(lwA)), rng.choice(lwB, n))
                    for _ in range(n_boot // 4)])
    out["logc_bridge"] = (lc, lcb.std(ddof=1), *np.percentile(lcb, [2.5, 97.5]))

    # 6. KL estimates, raw and corrected by the bridge log c
    out["KL_pθ||p_raw_corr"] = (-lwB.mean(), -(lwB.mean() - lc))
    out["KL_p||pθ_raw_corr"] = (lwA.mean(), lwA.mean() - lc)
    return out


def print_normalization_checks():
    for m in MODELS:
        print(f"\nModel {OUTPUT_LABEL[m]}")
        for k, v in normalization_checks(load_logliks(m)).items():
            if isinstance(v, tuple):
                print(f"  {k:<28} " + "  ".join(f"{x:9.4f}" for x in v))
            else:
                print(f"  {k:<28} {v:9.4f}")
    
# ════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Writing figures to: {FIG_DIR}\n")

    for m in MODELS:
        print(f"Model {OUTPUT_LABEL[m]} (data: {m}):")
        plot_sample_grid(m)
        plot_cluster_overlay(m)
        plot_pixel_proportion(m)
        plot_cluster_histogram(m)
        plot_loglik_histograms(m)
        plot_loglik_scatter(m)
        plot_loglik_scatter_combined(m)
        plot_ratio_histograms(m)
        plot_training_loss(m)

    print("\nSummary statistics:")
    print_summary_stats()

    print("\nDone.")
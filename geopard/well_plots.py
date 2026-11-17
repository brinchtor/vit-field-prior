"""
well_plots.py

Makes all figures for the well experiment from the files written by well_experiment.py.
Run from the same `code` folder (sharing its parent with `data`, `figures` and the checkpoint folder):

    python well_plots.py

Reads   data/well_experiment_<CKPT_NAME>/   (example_two_wells.npz, depth_scan.npz, ground_truth/*.npz,
                                              unconditional_samples.npz or unconditional_parts/*.npz)
        data/fluvial_binary_64x64.h5          (for the training mean)
Writes  figures/well_<CKPT_NAME>/*.png  and  summary.txt   (a new subfolder; nothing else is touched)

Every figure is made independently, so a missing input only skips the figures that need it.
Environment: CKPT_NAME (default checkpoints_fluvial_64_rowonly_bigger), PARENT_DIR (default: parent of code/).

Figures
  01  two-well example: test image, unweighted and IS-weighted posterior mean
  02  two-well example: posterior draws (unweighted and importance-resampled)
  03  two-well example: errors against the held-out image and posterior uncertainty
  04  two-well example: importance weights
  05  depth scan: how many training images match a well of each depth
  06  ground truth test: best / median / worst example
  07  MSE of the estimators against the reference (bars)
  08  MSE against distance to the well
  09  true vs ViT update effect
  10  per-cell squared error, prior vs ViT posterior, coloured by distance
  11  mean maps: reference, ViT posterior, error and MSE maps
  12  MSE by row (depth) and by distance from the well column
  13  ESS diagnostics
  14  per-test paired comparison
  15  decomposition into prior bias and update error
  16  update effect maps for a typical test
  17  calibration of the ViT posterior against the reference
"""

import os
import glob
import traceback
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

CKPT_NAME  = os.environ.get("CKPT_NAME", "checkpoints_fluvial_64_rowonly_bigger")
N_TEST_SET = 1000

parent_dir = os.environ.get("PARENT_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir   = os.path.join(parent_dir, "data")
run_dir    = os.path.join(data_dir, f"well_experiment_{CKPT_NAME}")
h5_path    = os.path.join(data_dir, "fluvial_binary_64x64.h5")
fig_dir    = os.path.join(parent_dir, "figures", f"well_{CKPT_NAME}")
os.makedirs(fig_dir, exist_ok=True)

COL_TRAIN  = "#1f5fbf"      # training mean (true prior mean, ignores the data)
COL_VPRIOR = "#6b6b6b"      # ViT unconditional mean (ignores the data)
COL_VIT    = "#d9730d"      # ViT posterior mean
COL_WELL   = "#d62728"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.dpi": 160,
})

summary = []


def say(line=""):
    print(line, flush=True)
    summary.append(line)


def save(fig, name):
    path = os.path.join(fig_dir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}", flush=True)


def image_axes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def mark_columns(ax, cols, alpha=0.35):
    for c in cols:
        ax.axvspan(c - 0.5, c + 0.5, color=COL_WELL, alpha=alpha, lw=0)


def mark_well(ax, col, depth, H, W):
    ax.plot([col, col], [-0.5, depth - 0.5], color=COL_WELL, lw=2.4, alpha=0.85, solid_capstyle="butt")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)


def run(name, fn):
    print(f"{name}", flush=True)
    try:
        fn()
    except Exception as e:                      # noqa: BLE001
        print(f"  SKIPPED ({type(e).__name__}: {e})", flush=True)
        if os.environ.get("DEBUG"):
            traceback.print_exc()


# ══ Load ═════════════════════════════════════════════════════════════════════
def load_npz(name):
    p = os.path.join(run_dir, name)
    return dict(np.load(p)) if os.path.exists(p) else None


EX = load_npz("example_two_wells.npz")
DS = load_npz("depth_scan.npz")

gt_files = sorted(glob.glob(os.path.join(run_dir, "ground_truth", "test_[0-9][0-9][0-9][0-9].npz")))
if gt_files:
    zs = [np.load(f) for f in gt_files]
    GT = {k: np.stack([z[k] for z in zs]) for k in zs[0].files}
elif os.path.exists(os.path.join(run_dir, "ground_truth_all.npz")):
    GT = dict(np.load(os.path.join(run_dir, "ground_truth_all.npz")))
else:
    GT = None

p_true = None
if os.path.exists(h5_path):
    with h5py.File(h5_path, "r") as f:
        ds = f["images"]
        n_train = ds.shape[0] - N_TEST_SET
        acc = np.zeros(ds.shape[1:])
        for s in range(0, n_train, 5000):
            acc += (ds[s:min(s + 5000, n_train)] > 0.5).sum(axis=0)
    p_true = acc / n_train

unc = None
unc_path = os.environ.get("UNCOND_PATH", os.path.join(data_dir, "generated_rowonly_bigger", "rowonly_bigger_epoch_0800_n1000.npz"))
if os.path.exists(unc_path):
    z = np.load(unc_path)
    key = "images" if "images" in z.files else next(k for k in z.files if z[k].ndim == 3)
    unc = z[key]
    if unc.ndim == 4:
        unc = unc[:, 0]
    print(f"ViT unconditional samples: {unc_path} (key '{key}', shape {unc.shape})", flush=True)
else:
    print(f"No unconditional samples at {unc_path}", flush=True)
p_vit = (unc > 0.5).mean(axis=0).astype(np.float64) if unc is not None else None
HAVE_PRIOR = p_vit is not None and p_true is not None

print(f"example: {EX is not None}, depth scan: {DS is not None}, ground truth: "
      f"{0 if GT is None else len(GT['ref_mean'])} tests, ViT prior samples: {0 if unc is None else len(unc)}, "
      f"training mean: {p_true is not None}", flush=True)
print(f"figures go to {fig_dir}\n", flush=True)

# ── Ground-truth derived quantities ──────────────────────────────────────────
if GT is not None:
    ref   = GT["ref_mean"].astype(np.float64)
    vit   = GT["vit_mean_is"].astype(np.float64)
    unw   = GT["vit_mean_unw"].astype(np.float64)
    nm    = GT["n_match"].astype(np.float64)
    floor = GT["ref_var_floor"].astype(np.float64)
    ess   = GT["ess"].astype(np.float64)
    cidx  = GT["cond_indices"]
    N, H, W = ref.shape
    depth    = int(cidx.shape[1])
    well_col = int(cidx[0][0] % W)

    free = np.ones((N, H * W), dtype=bool)
    for i in range(N):
        free[i, cidx[i]] = False
    free = free.reshape(N, H, W)

    rr, cc = np.mgrid[0:H, 0:W]
    dist = np.empty((N, H, W))
    for i in range(N):
        r0, c0 = cidx[i] // W, cidx[i] % W
        dist[i] = np.sqrt((rr[None] - r0[:, None, None]) ** 2 + (cc[None] - c0[:, None, None]) ** 2).min(axis=0)

    def per_test(se):
        return np.array([se[i][free[i]].mean() for i in range(N)])

    def se_of(a):
        return float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float("nan")

    se_vit = (vit - ref) ** 2
    se_unw = (unw - ref) ** 2
    m_vit = per_test(se_vit) - floor
    m_unw = per_test(se_unw) - floor
    if p_true is not None:
        se_train = (p_true[None] - ref) ** 2
        m_train = per_test(se_train) - floor
    if HAVE_PRIOR:
        se_vprior = (p_vit[None] - ref) ** 2
        m_vprior = per_test(se_vprior) - floor
        d_true = ref - p_true[None]
        d_vit = vit - p_vit[None]
        err_prior = np.broadcast_to(p_vit - p_true, ref.shape)
        err_upd = d_vit - d_true

    # ── Printed summary (also written to summary.txt) ────────────────────────
    say(f"Run: {CKPT_NAME}")
    say(f"Well: column {well_col}, rows 0-{depth - 1} (depth {depth} pixels), {N} test images")
    say(f"Matching training images: median {int(np.median(nm))}, min {int(nm.min())}, max {int(nm.max())}")
    say(f"Share of sand among observed pixels: "
        f"{np.mean([(GT['test_image'][i].reshape(-1)[cidx[i]] > 0.5).mean() for i in range(N)]) * 100:.0f}%")
    say(f"ESS: median {np.median(ess):.0f}, min {ess.min():.0f}, max {ess.max():.0f}")
    say("")
    say("MSE against the reference posterior mean, unobserved pixels, corrected for the reference's own Monte Carlo error")
    say(f"{'estimate':<28} {'MSE x1e-3':>10} {'s.e.':>7} {'wins vs ViT IS':>15}")
    rows = []
    if p_true is not None:
        rows.append(("training mean", m_train))
    if HAVE_PRIOR:
        rows.append(("ViT unconditional mean", m_vprior))
    rows.append(("ViT conditional, unweighted", m_unw))
    rows.append(("ViT conditional, IS", m_vit))
    for lab, m in rows:
        win = f"{(m_vit < m).mean() * 100:.0f}%" if lab != "ViT conditional, IS" else ""
        say(f"{lab:<28} {m.mean() * 1e3:>10.2f} {se_of(m) * 1e3:>7.2f} {win:>15}")
    if HAVE_PRIOR:
        mt = (d_true[free] ** 2).mean()
        mu = (err_upd[free] ** 2).mean()
        corr = float(np.corrcoef(d_true[free], d_vit[free])[0, 1])
        slope = float((d_true[free] * d_vit[free]).sum() / (d_true[free] ** 2).sum())
        say("")
        say(f"Size of the true update E[(p(x|d)-p(x))^2]: {mt * 1e3:.2f}e-3  (= MSE of the training mean)")
        say(f"Update skill 1 - E[(dV-dT)^2]/E[dT^2]: {1 - mu / mt:.2f}   corr {corr:.2f}   slope {slope:.2f}")
        say(f"Decomposition of the IS MSE: prior bias {(err_prior[free] ** 2).mean() * 1e3:.2f}e-3, "
            f"update error {mu * 1e3:.2f}e-3, cross {2 * (err_prior[free] * err_upd[free]).mean() * 1e3:+.2f}e-3")
    if EX is not None:
        say("")
        say(f"Two-well example: test image {int(EX['test_idx'])}, columns {[int(c) for c in EX['cond_cols']]}, "
            f"{len(EX['weights'])} samples, ESS {float(EX['ess']):.0f}")
    if DS is not None:
        say(f"Depth scan: chosen depth {int(DS['chosen_depth'])} (largest depth where >= {float(DS['pass_frac']):.0%} "
            f"of test images have >= {int(DS['min_matches'])} matches)")
    with open(os.path.join(fig_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")
    print()


# ══ 01-04: two-well example ═══════════════════════════════════════════════════
def fig01():
    img, cols = EX["test_image"], [int(c) for c in EX["cond_cols"]]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.5), constrained_layout=True)
    panels = [(img, f"Test image\n(columns {cols} observed)"),
              (EX["vit_mean_unw"], "Unweighted posterior mean"),
              (EX["vit_mean_is"], f"IS-weighted posterior mean\n(ESS = {float(EX['ess']):.0f})")]
    for ax, (m, t) in zip(axes, panels):
        ax.imshow(m, cmap="gray", vmin=0, vmax=1)
        mark_columns(ax, cols)
        ax.set_title(t)
        image_axes(ax)
    save(fig, "01_example_posterior_mean.png")


def fig02():
    S = EX["samples"].astype(np.float32)
    w = EX["weights"]
    cols = [int(c) for c in EX["cond_cols"]]
    rng = np.random.default_rng(0)
    n = 6
    unw_idx = np.arange(n)
    is_idx = rng.choice(len(S), size=n, replace=True, p=w / w.sum())
    fig = plt.figure(figsize=(14, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, n + 2, width_ratios=[2, 2] + [1] * n)
    ax = fig.add_subplot(gs[:, :2])
    ax.imshow(EX["test_image"], cmap="gray", vmin=0, vmax=1)
    mark_columns(ax, cols)
    ax.set_title("Held-out image\n(observed columns in red)")
    image_axes(ax)
    for r, (idx, lab) in enumerate([(unw_idx, "Unweighted draws"), (is_idx, "Importance-resampled draws")]):
        for j, k in enumerate(idx):
            a = fig.add_subplot(gs[r, 2 + j])
            a.imshow(S[k], cmap="gray", vmin=0, vmax=1)
            mark_columns(a, cols, alpha=0.3)
            image_axes(a)
            if j == 0:
                a.set_ylabel(lab, fontsize=9)
                a.yaxis.set_visible(True)
                a.set_yticks([])
                a.spines["left"].set_visible(False)
        fig.text(0.0, 0.0, "")
    fig.suptitle("Posterior samples given two wells", fontsize=11)
    save(fig, "02_example_posterior_draws.png")


def fig03():
    img = EX["test_image"]
    cols = [int(c) for c in EX["cond_cols"]]
    prior = p_vit if p_vit is not None else p_true
    prior_lab = "ViT unconditional mean" if p_vit is not None else "training mean"
    post = EX["vit_mean_is"]
    std = np.sqrt(np.clip(post * (1 - post), 0, None))
    free_cols = np.ones(img.shape[1], dtype=bool)
    free_cols[cols] = False
    se_prior = (prior - img) ** 2 if prior is not None else None
    se_post = (post - img) ** 2
    vmax = max(se_post.max(), se_prior.max() if se_prior is not None else 0)
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.4), constrained_layout=True)
    axes[0, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Held-out image")
    if prior is not None:
        axes[0, 1].imshow(prior, cmap="gray", vmin=0, vmax=1)
        axes[0, 1].set_title(f"Prior mean ({prior_lab})")
    axes[0, 2].imshow(post, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("IS-weighted posterior mean")
    if se_prior is not None:
        im = axes[1, 0].imshow(se_prior, cmap="magma_r", vmin=0, vmax=vmax)
        axes[1, 0].set_title(f"Squared error, prior mean\nMSE {se_prior[:, free_cols].mean():.4f} (unobserved columns)")
    im = axes[1, 1].imshow(se_post, cmap="magma_r", vmin=0, vmax=vmax)
    axes[1, 1].set_title(f"Squared error, posterior mean\nMSE {se_post[:, free_cols].mean():.4f} (unobserved columns)")
    im2 = axes[1, 2].imshow(std, cmap="viridis", vmin=0, vmax=0.5)
    axes[1, 2].set_title("Posterior uncertainty  sqrt(p(1-p))")
    for a in axes.ravel():
        mark_columns(a, cols, alpha=0.3)
        image_axes(a)
    fig.colorbar(im, ax=axes[1, :2], fraction=0.03, pad=0.02, label="squared error")
    fig.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.02, label="std")
    fig.suptitle("Two-well example: errors are measured against a single held-out image, not a posterior mean",
                 fontsize=10, color="0.35")
    save(fig, "03_example_errors_uncertainty.png")


def fig04():
    w = np.sort(EX["weights"])[::-1]
    ess_ = float(EX["ess"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), constrained_layout=True)
    axes[0].hist(np.log10(w), bins=40, color=COL_VIT, alpha=0.85)
    axes[0].set_xlabel("log10 normalised weight"); axes[0].set_ylabel("samples")
    axes[0].axvline(np.log10(1.0 / len(w)), color="0.3", ls="--", lw=1)
    axes[0].text(np.log10(1.0 / len(w)), axes[0].get_ylim()[1] * 0.95, " uniform", color="0.3", va="top", fontsize=8.5)
    axes[0].set_title("Weight distribution")
    axes[1].plot(np.arange(1, len(w) + 1), np.cumsum(w), color=COL_VIT, lw=2)
    axes[1].plot([0, len(w)], [0, 1], color="0.5", ls="--", lw=1)
    axes[1].set_xlabel("number of largest weights"); axes[1].set_ylabel("share of total weight")
    axes[1].set_title(f"Concentration (ESS = {ess_:.0f} of {len(w)})")
    lc, lu = EX["log_cond"], EX["log_uncond"]
    axes[2].scatter(lc, lu, s=8, color=COL_VIT, alpha=0.5)
    axes[2].set_xlabel("log q(x | y)  (proposal)"); axes[2].set_ylabel("log p(x)  (ViT prior)")
    axes[2].set_title("Log-likelihoods behind the weights")
    save(fig, "04_example_weights.png")


# ══ 05: depth scan ═══════════════════════════════════════════════════════════
def fig05():
    d, n = DS["depths"], DS["n_match"]
    chosen, mm, pf = int(DS["chosen_depth"]), int(DS["min_matches"]), float(DS["pass_frac"])
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    for thr, c in [(50, "#9ecae1"), (100, COL_TRAIN), (200, "#08306b")]:
        axes[0].plot(d, (n >= thr).mean(axis=0), color=c, lw=2, label=f">= {thr} matches")
    axes[0].axhline(pf, color="0.4", ls="--", lw=1)
    axes[0].axvline(chosen, color=COL_WELL, lw=1.5)
    axes[0].text(chosen + 0.5, 0.97, f" chosen depth {chosen}", color=COL_WELL, va="top", fontsize=9)
    axes[0].set_xlabel("well depth (pixels from the surface)"); axes[0].set_ylabel("share of test images")
    axes[0].set_title("Test images with enough matching training images"); axes[0].legend(loc="lower left")
    axes[0].set_ylim(0, 1.02)
    axes[1].plot(d, np.median(n, axis=0) + 0.5, color=COL_TRAIN, lw=2, label="median")
    axes[1].fill_between(d, np.percentile(n, 10, axis=0) + 0.5, np.percentile(n, 90, axis=0) + 0.5,
                         color=COL_TRAIN, alpha=0.15, lw=0, label="10-90 percentile")
    axes[1].axhline(mm, color="0.4", ls="--", lw=1)
    axes[1].axvline(chosen, color=COL_WELL, lw=1.5)
    axes[1].set_yscale("log"); axes[1].set_xlabel("well depth (pixels)"); axes[1].set_ylabel("matching training images")
    axes[1].set_title("Reference sample size"); axes[1].legend(loc="upper right")
    axes[2].hist(np.log10(n[:, chosen - 1] + 1), bins=40, color=COL_TRAIN, alpha=0.85)
    axes[2].axvline(np.log10(mm + 1), color="0.4", ls="--", lw=1)
    axes[2].set_xlabel("log10 (1 + matching training images)"); axes[2].set_ylabel("test images")
    axes[2].set_title(f"At the chosen depth ({chosen})")
    save(fig, "05_depth_scan.png")


# ══ 06-17: ground-truth test ═════════════════════════════════════════════════
def fig06():
    order = np.argsort(m_vit)
    picks = [("best", order[0]), ("median", order[N // 2]), ("worst", order[-1])]
    fig, axes = plt.subplots(len(picks), 4, figsize=(15.5, 3.9 * len(picks)), constrained_layout=True)
    for r, (name, i) in enumerate(picks):
        err = vit[i] - ref[i]
        v = max(np.abs(err).max(), 1e-6)
        axes[r, 0].imshow(GT["test_image"][i], cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_title(f"{name}: test image {int(GT['test_idx'][i])}")
        axes[r, 1].imshow(ref[i], cmap="gray", vmin=0, vmax=1)
        axes[r, 1].set_title(f"reference mean ({int(nm[i])} matches)")
        axes[r, 2].imshow(vit[i], cmap="gray", vmin=0, vmax=1)
        axes[r, 2].set_title(f"ViT IS mean (ESS {ess[i]:.0f})\nMSE {m_vit[i] * 1e3:.2f}e-3")
        im = axes[r, 3].imshow(err, cmap="RdBu_r", vmin=-v, vmax=v)
        axes[r, 3].set_title("ViT IS - reference")
        fig.colorbar(im, ax=axes[r, 3], fraction=0.046, pad=0.02)
        for a in axes[r]:
            mark_well(a, well_col, depth, H, W)
            image_axes(a)
    save(fig, "06_gt_examples.png")


def fig07():
    labs, vals, errs, cols = [], [], [], []
    if p_true is not None:
        labs.append("Training mean\n(ignores data)"); vals.append(m_train); cols.append(COL_TRAIN)
    if HAVE_PRIOR:
        labs.append("ViT unconditional\nmean (ignores data)"); vals.append(m_vprior); cols.append(COL_VPRIOR)
    labs.append("ViT conditional\nunweighted"); vals.append(m_unw); cols.append("#f0b27a")
    labs.append("ViT conditional\nIS-weighted"); vals.append(m_vit); cols.append(COL_VIT)
    means = np.array([v.mean() for v in vals]) * 1e3
    ses = np.array([se_of(v) for v in vals]) * 1e3
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    x = np.arange(len(vals))
    ax.bar(x, means, yerr=ses, color=cols, width=0.62, capsize=4, error_kw=dict(lw=1.2, ecolor="0.25"))
    for xi, m in zip(x, means):
        ax.text(xi, m * 0.5, f"{m:.2f}", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(r"MSE against reference ($\times10^{-3}$)")
    ax.set_title(f"Well of depth {depth}, {N} test images (error bars: 1 s.e.)")
    save(fig, "07_mse_bars.png")


def fig08():
    dedges = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]
    bins = []
    for lo, hi in zip(dedges[:-1], dedges[1:]):
        m = free & (dist >= lo) & (dist < hi)
        if m.sum() < 200:
            continue
        row = {"cen": float(np.median(dist[m]))}
        series = [("vit", se_vit)]
        if p_true is not None:
            series.append(("train", se_train))
        if HAVE_PRIOR:
            series.append(("vprior", se_vprior))
        for name, se in series:
            pt = np.array([se[i][m[i]].mean() for i in range(N) if m[i].any()])
            row[name] = float(se[m].mean())
            row[name + "_se"] = se_of(pt)
        bins.append(row)
    x = np.array([b["cen"] for b in bins])
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    spec = [("train", COL_TRAIN, "-", "training mean"), ("vprior", COL_VPRIOR, "--", "ViT unconditional mean"),
            ("vit", COL_VIT, "-", "ViT posterior mean (IS)")]
    for name, c, ls, lab in spec:
        if name not in bins[0]:
            continue
        y = np.array([b[name] for b in bins]); e = np.array([b[name + "_se"] for b in bins])
        ax.plot(x, y, ls, color=c, lw=2, marker="o", ms=4.5, label=lab)
        ax.fill_between(x, y - e, y + e, color=c, alpha=0.15, lw=0)
    noise = float(np.mean(floor))
    ax.axhline(noise, color="0.55", lw=1, ls=":")
    ax.text(x[-1], noise * 0.8, "reference Monte Carlo error", ha="right", va="top", color="0.4", fontsize=8.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("distance to nearest observed pixel"); ax.set_ylabel("MSE against reference posterior mean")
    ax.legend(loc="lower right")
    save(fig, "08_mse_vs_distance.png")


def fig09():
    x, y = d_true[free], d_vit[free]
    lim = float(np.ceil(np.percentile(np.abs(np.concatenate([x, y])), 99.9) * 10) / 10)
    lim = max(lim, 0.2)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    hb = axes[0].hexbin(x, y, gridsize=70, bins="log", mincnt=1, cmap="Blues", extent=(-lim, lim, -lim, lim))
    axes[0].plot([-lim, lim], [-lim, lim], color="0.25", lw=0.8, ls="--")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel(r"true update  $p(x\,|\,d)-p(x)$"); axes[0].set_ylabel(r"ViT update  $p^*(x\,|\,d)-p^*(x)$")
    axes[0].set_title(f"Update effect (corr {np.corrcoef(x, y)[0, 1]:.2f}, slope {(x * y).sum() / (x * x).sum():.2f})")
    fig.colorbar(hb, ax=axes[0], fraction=0.046, pad=0.02, label="cells")
    edges = [1, 2, 3, 5, 9, 17, 33, 1e9]
    labels = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33+"]
    cors, slopes, sizes_t, sizes_v, xs = [], [], [], [], []
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = free & (dist >= lo) & (dist < hi)
        if m.sum() < 200:
            continue
        a, b = d_true[m], d_vit[m]
        cors.append(np.corrcoef(a, b)[0, 1]); slopes.append((a * b).sum() / (a * a).sum())
        sizes_t.append(np.abs(a).mean()); sizes_v.append(np.abs(b).mean()); xs.append(labels[j])
    axes[1].plot(xs, sizes_t, "o-", color=COL_TRAIN, lw=2, label="truth")
    axes[1].plot(xs, sizes_v, "o-", color=COL_VIT, lw=2, label="ViT")
    axes[1].set_ylabel(r"mean $|$update$|$"); axes[1].set_xlabel("distance to nearest observed pixel")
    axes[1].set_title("Size of the update effect"); axes[1].legend()
    axes[2].plot(xs, cors, "o-", color="0.25", lw=2, label="correlation")
    axes[2].plot(xs, slopes, "o--", color="0.55", lw=2, label="slope through origin")
    axes[2].axhline(1, color="0.7", lw=0.8)
    axes[2].set_xlabel("distance to nearest observed pixel"); axes[2].set_title("Agreement with the true update")
    axes[2].legend()
    save(fig, "09_update_effect.png")


def fig10():
    N_PLOT, FLOOR = 60000, 1e-6
    rng = np.random.default_rng(0)
    sel = np.flatnonzero(free.ravel())
    sel = rng.choice(sel, size=min(N_PLOT, len(sel)), replace=False)
    sel = sel[np.argsort(-dist.ravel()[sel])]
    dsel = dist.ravel()[sel]
    y = np.clip(se_vit.ravel()[sel], FLOOR, None)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), constrained_layout=True)
    sc = None
    for a, se_p, ttl in zip(axes, [se_train, se_vprior], ["prior = training mean", "prior = ViT unconditional mean"]):
        x = np.clip(se_p.ravel()[sel], FLOOR, None)
        sc = a.scatter(x, y, c=dsel, s=3, alpha=0.5, cmap="viridis", norm=LogNorm(vmin=1, vmax=max(dsel.max(), 2)),
                       rasterized=True)
        a.plot([FLOOR, 1], [FLOOR, 1], color="0.2", lw=0.8, ls="--")
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(FLOOR, 1); a.set_ylim(FLOOR, 1)
        a.set_aspect("equal")
        a.set_xlabel("squared error of the prior mean vs reference"); a.set_ylabel("squared error of the ViT posterior mean")
        a.set_title(f"{ttl}\nshare below diagonal: {(y < x).mean() * 100:.0f}%")
    fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02, label="distance to nearest observed pixel")
    save(fig, "10_per_cell_squared_error.png")


def fig11():
    mse_maps = [("training mean", ((p_true[None] - ref) ** 2).mean(axis=0))] if p_true is not None else []
    if HAVE_PRIOR:
        mse_maps.append(("ViT unconditional mean", ((p_vit[None] - ref) ** 2).mean(axis=0)))
    mse_maps.append(("ViT posterior mean", ((vit - ref) ** 2).mean(axis=0)))
    vmax = max(m.max() for _, m in mse_maps)
    n_top = 3
    fig, axes = plt.subplots(2, max(n_top, len(mse_maps)), figsize=(5 * max(n_top, len(mse_maps)), 9),
                             constrained_layout=True)
    bias = (vit - ref).mean(axis=0)
    vb = max(np.abs(bias).max(), 1e-6)
    top = [(ref.mean(axis=0), "gray", 0, 1, "Mean reference posterior mean"),
           (vit.mean(axis=0), "gray", 0, 1, "Mean ViT posterior mean"),
           (bias, "RdBu_r", -vb, vb, "Mean (ViT - reference)")]
    for ax, (m, cm, lo, hi, t) in zip(axes[0], top):
        im = ax.imshow(m, cmap=cm, vmin=lo, vmax=hi)
        ax.set_title(t)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for ax, (t, m) in zip(axes[1], mse_maps):
        im = ax.imshow(m, cmap="magma_r", vmin=0, vmax=vmax)
        ax.set_title(f"MSE map: {t}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for ax in axes.ravel():
        if ax.images:
            mark_well(ax, well_col, depth, H, W)
            image_axes(ax)
        else:
            ax.axis("off")
    save(fig, "11_mean_and_error_maps.png")


def fig12():
    def profile_rows(se):
        return np.array([se[:, r, :][free[:, r, :]].mean() for r in range(H)])

    def profile_cols(se):
        dc = np.abs(np.arange(W) - well_col)
        out = []
        for d in range(int(dc.max()) + 1):
            cols_ = np.flatnonzero(dc == d)
            m = free[:, :, cols_]
            out.append(se[:, :, cols_][m].mean() if m.any() else np.nan)
        return np.array(out)

    ests = [("vit", se_vit, COL_VIT, "-", "ViT posterior mean")]
    if p_true is not None:
        ests.insert(0, ("train", se_train, COL_TRAIN, "-", "training mean"))
    if HAVE_PRIOR:
        ests.insert(1 if p_true is not None else 0, ("vprior", se_vprior, COL_VPRIOR, "--", "ViT unconditional mean"))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), constrained_layout=True)
    for _, se, c, ls, lab in ests:
        axes[0].plot(np.arange(H), profile_rows(se), ls, color=c, lw=2, label=lab)
        pc = profile_cols(se)
        axes[1].plot(np.arange(len(pc)), pc, ls, color=c, lw=2, label=lab)
    axes[0].axvspan(0, depth - 1, color=COL_WELL, alpha=0.08, lw=0)
    axes[0].text(depth / 2, axes[0].get_ylim()[1] * 0.97, "well", color=COL_WELL, ha="center", va="top", fontsize=9)
    axes[0].set_xlabel("row (depth, pixels from the surface)"); axes[0].set_ylabel("MSE against reference")
    axes[0].set_title("MSE by depth"); axes[0].legend()
    axes[1].set_xlabel("distance from the well column (pixels)"); axes[1].set_ylabel("MSE against reference")
    axes[1].set_title("MSE by lateral distance from the well")
    save(fig, "12_mse_profiles.png")


def fig13():
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    axes[0].hist(ess, bins=25, color=COL_VIT, alpha=0.85)
    axes[0].set_xlabel("importance-weight ESS"); axes[0].set_ylabel("test images")
    axes[0].set_title(f"ESS (median {np.median(ess):.0f} of {GT['vit_mean_unw'].shape[0] and int(round(ess.max()))} max)")
    axes[1].scatter(nm, ess, s=16, color=COL_VIT, alpha=0.8)
    axes[1].set_xscale("log"); axes[1].set_xlabel("matching training images"); axes[1].set_ylabel("ESS")
    axes[1].set_title("ESS vs reference sample size")
    axes[2].scatter(ess, m_vit * 1e3, s=16, color=COL_VIT, alpha=0.8)
    axes[2].set_xlabel("ESS"); axes[2].set_ylabel(r"MSE of ViT IS mean ($\times10^{-3}$)")
    axes[2].set_title("Error vs ESS")
    save(fig, "13_ess_diagnostics.png")


def fig14():
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)
    comp = []
    if p_true is not None:
        comp.append((m_train, "training mean", COL_TRAIN))
    if HAVE_PRIOR:
        comp.append((m_vprior, "ViT unconditional mean", COL_VPRIOR))
    for ax, (m, lab, c) in zip(axes[:2], comp):
        lim = max(m.max(), m_vit.max()) * 1.05
        ax.scatter(m * 1e3, m_vit * 1e3, s=18, color=c, alpha=0.8)
        ax.plot([0, lim * 1e3], [0, lim * 1e3], color="0.3", ls="--", lw=0.9)
        ax.set_xlim(0, lim * 1e3); ax.set_ylim(0, lim * 1e3); ax.set_aspect("equal")
        ax.set_xlabel(rf"MSE, {lab} ($\times10^{{-3}}$)"); ax.set_ylabel(r"MSE, ViT posterior mean ($\times10^{-3}$)")
        ax.set_title(f"ViT posterior better in {(m_vit < m).mean() * 100:.0f}% of tests")
    if p_true is not None:
        diff = (m_vit - m_train) * 1e3
        axes[2].hist(diff, bins=25, color=COL_VIT, alpha=0.85)
        axes[2].axvline(0, color="0.3", lw=1)
        axes[2].axvline(diff.mean(), color=COL_TRAIN, lw=1.5)
        axes[2].set_xlabel(r"MSE(ViT posterior) - MSE(training mean)  ($\times10^{-3}$)"); axes[2].set_ylabel("test images")
        axes[2].set_title(f"mean difference {diff.mean():+.2f} (s.e. {se_of(diff):.2f})")
    save(fig, "14_per_test_paired.png")


def fig15():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    bias_map = p_vit - p_true
    v = np.abs(bias_map).max()
    im = axes[0].imshow(bias_map, cmap="RdBu_r", vmin=-v, vmax=v)
    axes[0].set_title(f"Prior bias  p* - p   (MSE {(bias_map ** 2).mean() * 1e3:.2f}e-3)")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02)
    upd = np.nanmean(np.where(free, err_upd, np.nan), axis=0)
    v = np.nanmax(np.abs(upd))
    im = axes[1].imshow(upd, cmap="RdBu_r", vmin=-v, vmax=v)
    mark_well(axes[1], well_col, depth, H, W)
    axes[1].set_title("Mean update error  (dV - dT)")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
    for a in axes[:2]:
        image_axes(a)
    mu = np.array([(err_upd[i][free[i]] ** 2).mean() for i in range(N)])
    mt = np.array([(d_true[i][free[i]] ** 2).mean() for i in range(N)])
    mp = float((bias_map ** 2).mean())
    lo, hi = min(mu.min(), mt.min(), mp) * 0.7, max(mu.max(), mt.max(), mp) * 1.4
    axes[2].scatter(mt, mu, s=16, color=COL_VIT, alpha=0.8)
    axes[2].plot([lo, hi], [lo, hi], color="0.3", lw=0.8, ls="--")
    axes[2].axhline(mp, color=COL_TRAIN, lw=1.5)
    axes[2].text(lo * 1.1, mp * 1.08, "prior bias MSE (same for all tests)", color=COL_TRAIN, va="bottom", fontsize=8.5)
    axes[2].set_xscale("log"); axes[2].set_yscale("log"); axes[2].set_xlim(lo, hi); axes[2].set_ylim(lo, hi)
    axes[2].set_xlabel("size of the true update  E[dT^2]"); axes[2].set_ylabel("update error  E[(dV - dT)^2]")
    axes[2].set_title("Per test (below the dashed line: ViT update beats no update)")
    save(fig, "15_prior_bias_vs_update_error.png")


def fig16():
    skill = np.array([1 - ((d_vit[i][free[i]] - d_true[i][free[i]]) ** 2).mean() / (d_true[i][free[i]] ** 2).mean()
                      for i in range(N)])
    i = int(np.argsort(skill)[N // 2])
    dt = np.where(free[i], d_true[i], np.nan)
    dv = np.where(free[i], d_vit[i], np.nan)
    v = np.nanmax(np.abs(np.concatenate([dt.ravel(), dv.ravel()])))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for a, m, t in zip(axes, [dt, dv, dv - dt],
                       ["true update  p(x|d) - p(x)", "ViT update  p*(x|d) - p*(x)", "ViT update - true update"]):
        im = a.imshow(m, cmap="RdBu_r", vmin=-v, vmax=v)
        mark_well(a, well_col, depth, H, W)
        a.set_title(t)
        image_axes(a)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    fig.suptitle(f"Test image {int(GT['test_idx'][i])} (median update skill {skill[i]:.2f})", fontsize=10.5)
    save(fig, "16_update_effect_example.png")


def fig17():
    xs = ref[free].ravel()
    bins = np.linspace(0, 1, 11)
    which = np.clip(np.digitize(xs, bins) - 1, 0, 9)
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, ax = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color="0.5", ls="--", lw=1)
    series = [("ViT posterior mean", vit, COL_VIT, "-")]
    if p_true is not None:
        series.append(("training mean", np.broadcast_to(p_true[None], ref.shape), COL_TRAIN, "-"))
    if HAVE_PRIOR:
        series.append(("ViT unconditional mean", np.broadcast_to(p_vit[None], ref.shape), COL_VPRIOR, "--"))
    for lab, arr, c, ls in series:
        ys = arr[free].ravel()
        mean_y = [ys[which == b].mean() if (which == b).any() else np.nan for b in range(10)]
        ax.plot(centers, mean_y, ls, color=c, lw=2, marker="o", ms=4.5, label=lab)
    ax.set_xlabel("reference posterior probability (binned)"); ax.set_ylabel("mean estimated probability")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(loc="upper left")
    save(fig, "17_calibration.png")


if EX is not None:
    run("01 two-well example, posterior means", fig01)
    run("02 two-well example, posterior draws", fig02)
    run("03 two-well example, errors and uncertainty", fig03)
    run("04 two-well example, importance weights", fig04)
if DS is not None:
    run("05 depth scan", fig05)
if GT is not None:
    run("06 ground truth, best / median / worst", fig06)
    run("07 MSE bars", fig07)
    run("08 MSE vs distance", fig08)
    if HAVE_PRIOR:
        run("09 update effect", fig09)
        run("10 per-cell squared error", fig10)
    run("11 mean and error maps", fig11)
    run("12 MSE profiles", fig12)
    run("13 ESS diagnostics", fig13)
    run("14 per-test paired comparison", fig14)
    if HAVE_PRIOR:
        run("15 prior bias vs update error", fig15)
        run("16 update effect example", fig16)
    run("17 calibration", fig17)
else:
    print("No ground-truth files found, skipping figures 06-17.")

print(f"\nDone. Figures and summary.txt are in {fig_dir}")

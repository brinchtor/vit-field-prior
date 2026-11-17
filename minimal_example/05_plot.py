"""
05_plot.py

Summarise the minimal example:
  fig_loglik.png    ViT log p_theta(x) vs exact log p(x) on held-out images
  fig_posterior.png per observed image: truth + observed pixels, exact
                    P(x=1|y), ViT IS-weighted and unweighted estimates, IS - exact
  summary.txt       marginal MAE vs exact, ESS, log-likelihood statistics

Output: out/fig_loglik.png, out/fig_posterior.png, out/summary.txt
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from run_all import S, out  # noqa: E402

ll = np.load(out("logliks_A.npz"))
exact, vit = ll["exact"], ll["vit"]
lines = [f"log-likelihood, {len(exact)} held-out images: mean exact = {exact.mean():.3f}, "
         f"mean ViT = {vit.mean():.3f}, corr = {np.corrcoef(exact, vit)[0, 1]:.3f}"]

fig, ax = plt.subplots(figsize=(4.2, 4))
lo, hi = min(exact.min(), vit.min()), max(exact.max(), vit.max())
ax.plot([lo, hi], [lo, hi], color="0.6", lw=1, ls="--", zorder=1)
ax.scatter(exact, vit, s=12, color="#2a6fb0", alpha=0.7, linewidths=0, zorder=2)
ax.set_xlabel("exact log p(x)")
ax.set_ylabel("ViT log p_theta(x)")
ax.set_title("Model A, held-out samples (dashed: y = x)", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(out("fig_loglik.png"), dpi=150)
plt.close(fig)

rows = []
for regime in ["random", "wells"]:
    obs = np.load(out(f"obs_A_{regime}.npz"))
    ex = np.load(out(f"exact_A_{regime}.npz"))["marginals"][..., 1]
    v = np.load(out(f"vit_A_{regime}.npz"))
    mae_is = np.abs(v["mean_weighted"] - ex).mean(axis=(1, 2))
    mae_un = np.abs(v["mean_unweighted"] - ex).mean(axis=(1, 2))
    lines.append(f"{regime}: marginal MAE vs exact  IS = {np.round(mae_is, 4).tolist()}, "
                 f"unweighted = {np.round(mae_un, 4).tolist()}, ESS = {np.round(v['ess'], 1).tolist()} of S={S}")
    for i in range(len(ex)):
        rows.append((f"{regime} {i}", obs["images"][i], obs["obs_rows"][i], obs["obs_cols"][i],
                     ex[i], v["mean_weighted"][i], v["mean_unweighted"][i], v["ess"][i]))

titles = ["truth + observed", "exact P(x=1|y)", "ViT IS-weighted", "ViT unweighted", "IS - exact"]
fig, axes = plt.subplots(len(rows), 5, figsize=(12, 1.25 * len(rows) + 0.6), squeeze=False)
for r, (label, img, orow, ocol, ex_i, w_i, u_i, ess_i) in enumerate(rows):
    axes[r, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[r, 0].scatter(ocol, orow, s=6, facecolors="none", edgecolors="#e07b00", linewidths=0.8)
    for c, field in zip((1, 2, 3), (ex_i, w_i, u_i)):
        im_p = axes[r, c].imshow(field, cmap="Blues", vmin=0, vmax=1)
    im_d = axes[r, 4].imshow(w_i - ex_i, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axes[r, 0].set_ylabel(f"{label}\nESS {ess_i:.0f}", fontsize=8)
for c, t in enumerate(titles):
    axes[0, c].set_title(t, fontsize=9)
for a in axes.ravel():
    a.set_xticks([])
    a.set_yticks([])
fig.colorbar(im_p, ax=axes[:, 1:4], fraction=0.02, pad=0.01, label="P(x = 1)")
fig.colorbar(im_d, ax=axes[:, 4], fraction=0.05, pad=0.02, label="difference")
fig.savefig(out("fig_posterior.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

with open(out("summary.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))

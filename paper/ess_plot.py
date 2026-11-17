import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures_condexp", exist_ok=True)

MODELS = ["A", "B", "C", "D"]

for obs_type in ["wells", "random"]:
    fig, axes = plt.subplots(1, 4, figsize=(10, 3), sharey=True)

    for ax, model in zip(axes, MODELS):
        npz = np.load(f"data/cond_exp/vit_{model}_{obs_type}.npz")
        log_lik_cond   = npz["log_lik_cond"]
        log_lik_uncond = npz["log_lik_uncond"]
        N = log_lik_cond.shape[1]

        log_w = log_lik_uncond - log_lik_cond
        log_w -= log_w.max(axis=1, keepdims=True)
        w = np.exp(log_w)
        w /= w.sum(axis=1, keepdims=True)
        ess = 1.0 / (w ** 2).sum(axis=1)
        bins = np.arange(0, N + 51, 50)

        ax.hist(ess, bins=bins, density=True, color="darkorange", edgecolor="white")
        ax.axvline(N, color="black", linewidth=0.8, linestyle="--", alpha=0.6, label=f"$N={N}$")
        ax.set_xlabel("ESS")
        ax.set_title(f"Model {model}")
        ax.grid(True, linewidth=0.4)

    axes[0].set_ylabel("Density")
    plt.tight_layout()
    plt.savefig(f"figures_condexp/ess_all_{obs_type}.pdf", bbox_inches="tight")
    plt.savefig(f"figures_condexp/ess_all_{obs_type}.png", dpi=150, bbox_inches="tight")
    plt.close()

print("Done.")
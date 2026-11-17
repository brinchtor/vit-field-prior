"""
04_posterior.py

For every observed image:
  - exact posterior marginals P(x=1|y) with common/exact_marginals.marginals
    (as paper/eval_exact_A.py);
  - importance-sampling posterior with the ViT prior (as
    paper/conditional_model_A_*.py + paper/plot_cond_exp.py, S samples):
    x_s ~ q(x|y) via ViT_sampling_rectangle_GPU.generate_images_batch
    (observed pixels clamped, Manhattan order), w_s ~ p_theta(x_s)/q(x_s|y)
    (exact observations: p(y|x)=1), ESS = 1/sum w_s^2.

Output: out/exact_A_{random,wells}.npz (marginals, logZ)
        out/vit_A_{random,wells}.npz (samples, log_lik_cond, log_lik_uncond,
                                      weights, mean_weighted, mean_unweighted, ess)
"""
import numpy as np
import torch

from run_all import BET, N1, N2, PATCH_SIZE, S, alf, load_vit, out, seed_everything  # sets sys.path
import exact_marginals
from generate_mrf_datasets_A import load_obs_dicts
from ViT_sampling_rectangle_GPU import generate_images_batch

seed_everything()
model = load_vit()

for regime in ["random", "wells"]:
    margs, logZs = zip(*[exact_marginals.marginals(alf(), BET, observations=o, return_logZ=True)
                         for o in load_obs_dicts(out(f"obs_A_{regime}.npz"))])
    np.savez_compressed(out(f"exact_A_{regime}.npz"), marginals=np.array(margs), logZ=np.array(logZs))

    data = np.load(out(f"obs_A_{regime}.npz"))
    n = data["vit_indices"].shape[0]
    samples = np.empty((n, S, N1, N2), dtype=np.float32)
    lc, lu = np.empty((n, S)), np.empty((n, S))
    for i in range(n):
        imgs, log_cond, log_unc = generate_images_batch(
            model=model, patch_size=PATCH_SIZE, image_size=(N1, N2), batch_size=S,
            device=torch.device("cpu"), condition_indices=data["vit_indices"][i],
            condition_values=data["vit_values"][i], generation_order="manhattan",
        )
        samples[i], lc[i], lu[i] = imgs.numpy(), log_cond.numpy(), log_unc.numpy()

    log_w = lu - lc
    w = np.exp(log_w - log_w.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    ess = 1.0 / (w ** 2).sum(axis=1)
    np.savez_compressed(out(f"vit_A_{regime}.npz"), samples=samples, log_lik_cond=lc,
                        log_lik_uncond=lu, weights=w,
                        mean_weighted=(w[:, :, None, None] * samples).sum(axis=1),
                        mean_unweighted=samples.mean(axis=1), ess=ess)
    print(f"{regime}: ESS per image = {np.round(ess, 1).tolist()} (of S={S})")

"""
03_eval_loglik.py

ViT log-likelihood log p_theta(x) vs exact model-A log p(x) on held-out exact
samples, using mrf_log_unnormalized and vit_loglik from paper/eval_logliks.py
(raster reveal order, as in the paper).

Output: out/logliks_A.npz (exact, vit, logZ)
"""
import numpy as np
import torch

from run_all import BET, PATCH_SIZE, alf, load_vit, out, seed_everything
from eval_logliks import mrf_log_unnormalized, vit_loglik

seed_everything()
test = np.load(out("mrf_A_test.npz"))
imgs, logZ = test["images"].astype(np.int64), float(test["logZ"])

exact = mrf_log_unnormalized(imgs, alf(), BET) - logZ
vit = vit_loglik(load_vit(), imgs, PATCH_SIZE, "raster", 50, torch.device("cpu"))
np.savez_compressed(out("logliks_A.npz"), exact=exact, vit=vit, logZ=logZ)
print(f"mean exact log p(x) = {exact.mean():.3f}, mean ViT log p_theta(x) = {vit.mean():.3f}")
print(f"mean diff = {(vit - exact).mean():.3f} nats/image, corr = {np.corrcoef(exact, vit)[0, 1]:.3f}")

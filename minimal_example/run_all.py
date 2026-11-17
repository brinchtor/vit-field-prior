"""
run_all.py

Minimal, CPU-only example for model A (paper Model I: first-order binary
Ising/Potts MRF on a 12 x 24 grid, alf = 0, bet = 0.7). Runs in a few minutes.

Usage (from the repository root, after `pip install -e .`):
    python minimal_example/run_all.py

This file holds the shared settings imported by the five step scripts and,
when executed, runs them in order. Each step's full output is written to
minimal_example/out/logs/<step>.log; all results go to minimal_example/out/.
Sizes are deliberately tiny, so results are NOT comparable to the paper.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "out")
VIT_DIR = os.path.join(OUT, "vit_A")

# Reuse repository code: common/ (also importable after `pip install -e .`) and
# paper/ for generate_mrf_datasets_A and eval_logliks (their __main__ blocks
# are not executed on import).
for _p in (os.path.join(REPO, "common"), os.path.join(REPO, "paper")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Model A (as in paper/generate_samples_MRF.py) ────────────────────────────
N1, N2 = 12, 24
BET = 0.7

# ── Tiny sizes ───────────────────────────────────────────────────────────────
N_TRAIN = 2000       # exact samples, doubled to 4000 by label flipping
N_TEST = 200         # held-out exact samples for the log-likelihood check
N_OBS_IMAGES = 3     # ground-truth images per observation regime
N_RANDOM_OBS = 30    # "random" regime: pixels observed per image
WELL_COLS = [6, 18]  # "wells" regime: fully observed columns
S = 200              # ViT posterior samples per observed image (paper: 1000)

# ── ViT: paper/train_MRF_model_A.py architecture, far fewer epochs ──────────
PATCH_SIZE = 2
VIT_KW = dict(emb_dim=64, num_heads=2, num_layers=2, ffn_dim=256)
VOCAB_CAP = 16
EPOCHS = 8           # paper: 1000
BATCH_SIZE = 100
LEARNING_RATE = 1e-3
SEED = 0

STEPS = ["01_data", "02_train_vit", "03_eval_loglik", "04_posterior", "05_plot"]


def alf():
    import numpy as np
    return np.zeros((N1, N2, 2), dtype=float)


def out(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def seed_everything(seed=SEED):
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_vit():
    """Rebuild the ViT trained by 02_train_vit.py from out/vit_A/ (CPU)."""
    import torch
    import torch.nn as nn
    from ViT_rectangle import StackedContextViT

    vocab_ckpt = torch.load(os.path.join(VIT_DIR, "vocab_checkpoint.pt"), map_location="cpu")
    model = StackedContextViT(
        vocab=vocab_ckpt["vocab"],
        mask_token=nn.Parameter(vocab_ckpt["mask_token"], requires_grad=True),
        patch_dim=PATCH_SIZE * PATCH_SIZE,
        num_patches=(N1 // PATCH_SIZE) * (N2 // PATCH_SIZE),
        grid_size=(N1 // PATCH_SIZE, N2 // PATCH_SIZE),
        use_pos_emb=True,
        use_rel_bias=True,
        **VIT_KW,
    )
    model.load_state_dict(torch.load(os.path.join(VIT_DIR, "model_final.pth"), map_location="cpu"))
    model.eval()
    return model


def main():
    logs = os.path.join(OUT, "logs")
    os.makedirs(logs, exist_ok=True)
    start = time.time()
    for step in STEPS:
        t0 = time.time()
        print(f"==> {step}", flush=True)
        log_path = os.path.join(logs, f"{step}.log")
        with open(log_path, "w", encoding="utf-8") as log:
            rc = subprocess.call([sys.executable, os.path.join(HERE, f"{step}.py")],
                                 cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        lines = [l.rstrip() for l in open(log_path, encoding="utf-8", errors="replace")
                 if l.strip() and "it/s]" not in l and "s/it]" not in l]
        if rc != 0:
            print(f"FAILED: {step} (exit {rc}); last lines of {log_path}:")
            print("\n".join(lines[-20:]))
            sys.exit(1)
        print("\n".join("    " + l for l in lines[-4:]))
        print(f"    ({time.time() - t0:.0f} s)", flush=True)
    print(f"All steps finished in {time.time() - start:.0f} s. Outputs: minimal_example/out/")


if __name__ == "__main__":
    main()

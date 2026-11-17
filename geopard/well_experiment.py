"""
well_experiment.py

Run from a folder called `code` that shares its parent with `checkpoints_*`, `data` and `figures`:

    parent/
        code/          <- this script
        data/          <- fluvial_binary_64x64.h5, output goes here
        figures/
        checkpoints_fluvial_64_rowonly_bigger/   (vocab_checkpoint.pt, model_epoch_*.pth)

Stages (nothing is plotted; see well_plots.py for the figures):

  1. Two-well posterior sampling example. All 64 pixels of columns EXAMPLE_COLS (22 and 43) are observed
     on test image EXAMPLE_TEST_IDX, and N_EXAMPLE_SAMPLES posterior samples are drawn (Manhattan ordering).
     Saved to example_two_wells.npz (samples, log-likelihoods, weights, means, ESS).

  2. Depth scan. For a single vertical well in column WELL_COL, starting at the surface (row 0) and going
     down K rows, the number of training images that match the observed pixels exactly is counted for every
     test image and every depth K = 1..H. The chosen depth is the largest K for which at least PASS_FRAC of
     the test images are matched by at least MIN_MATCHES training images ("often a reasonable number of
     matches"). Set DEPTH=<int> to override the choice. Saved to depth_scan.npz.

  3. Ground-truth test with that one fixed well. The first N_TESTS test images that satisfy the match
     requirement are used. For each, the reference posterior mean is the mean of the matching training
     images, and N_SAMPLES ViT samples are drawn with GEN_ORDER (default Manhattan) ordering for the
     unobserved patches. Per-test files are written with the same fields as before (ref_mean, n_match,
     ref_var_floor, vit_mean_unw, vit_mean_is, ess, cond_indices, cond_values, test_idx, test_image) plus
     vit_std_is.

  4. Unconditional ViT samples from the same model (N_UNCOND, default 10000), needed for the "ViT prior
     mean" baseline. Written as parts so that array jobs can share the work.

Everything lands in data/well_experiment_<CKPT_NAME>/ :
    example_two_wells.npz
    depth_scan.npz
    ground_truth/test_XXXX.npz
    unconditional_parts/part_XXX.npz
    (after COMBINE_ONLY=1)  ground_truth_all.npz, unconditional_samples.npz

Parallel use (SLURM array): TASK_ID / N_TASKS (or SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT) split stages 3
and 4; stages 1 and 2 are done by task 0. Everything is resumable: finished files are skipped.
When all tasks are done:   COMBINE_ONLY=1 python well_experiment.py

Environment variables (defaults):
    CKPT_NAME=checkpoints_fluvial_64_rowonly_bigger   NUM_HEADS=4  FFN_DIM=512  PATCH_SIZE=4
    GEN_ORDER=manhattan   BATCH_SIZE=100
    EXAMPLE_COLS=22,43   EXAMPLE_TEST_IDX=0   N_EXAMPLE_SAMPLES=1000
    WELL_COL=32   MIN_MATCHES=100   PASS_FRAC=0.5   DEPTH=(auto)   N_TESTS=100   N_SAMPLES=2000
    N_UNCOND=10000   UNCOND_ORDER=(sampler default)   PARENT_DIR=(parent of this script's folder)
The embedding dimension and number of layers are read from the checkpoint.
"""

import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import h5py

from ViT_rowonly import *
from ViT_sampling_rowonly import *


def _int(name, default):
    return int(os.environ.get(name, default))


# ── Config ────────────────────────────────────────────────────────────────────
CKPT_NAME         = os.environ.get("CKPT_NAME", "checkpoints_fluvial_64_rowonly_bigger")
NUM_HEADS         = _int("NUM_HEADS", 4)
FFN_DIM           = _int("FFN_DIM", 512)
PATCH_SIZE        = _int("PATCH_SIZE", 4)
IMAGE_SIZE        = (64, 64)
N_TEST_SET        = 1000
GEN_ORDER         = os.environ.get("GEN_ORDER", "manhattan")
BATCH_SIZE        = _int("BATCH_SIZE", 100)
SEED              = 0

EXAMPLE_COLS      = [int(c) for c in os.environ.get("EXAMPLE_COLS", "22,43").split(",")]
EXAMPLE_TEST_IDX  = _int("EXAMPLE_TEST_IDX", 0)
N_EXAMPLE_SAMPLES = _int("N_EXAMPLE_SAMPLES", 1000)

WELL_COL          = _int("WELL_COL", 32)
MIN_MATCHES       = _int("MIN_MATCHES", 100)
PASS_FRAC         = float(os.environ.get("PASS_FRAC", 0.5))
DEPTH_OVERRIDE    = os.environ.get("DEPTH")
N_TESTS           = _int("N_TESTS", 100)
N_SAMPLES         = _int("N_SAMPLES", 2000)

N_UNCOND          = _int("N_UNCOND", 10000)
UNCOND_ORDER      = os.environ.get("UNCOND_ORDER")

# ── Paths ─────────────────────────────────────────────────────────────────────
parent_dir     = os.environ.get("PARENT_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
checkpoint_dir = os.path.join(parent_dir, CKPT_NAME)
data_dir       = os.path.join(parent_dir, "data")
data_path      = os.path.join(data_dir, "fluvial_binary_64x64.h5")
run_dir        = os.path.join(data_dir, f"well_experiment_{CKPT_NAME}")
gt_dir         = os.path.join(run_dir, "ground_truth")
parts_dir      = os.path.join(run_dir, "unconditional_parts")
for d in (run_dir, gt_dir, parts_dir):
    os.makedirs(d, exist_ok=True)

task_id = int(os.environ.get("TASK_ID", os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
n_tasks = int(os.environ.get("N_TASKS", os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))


def save_npz(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)          # atomic: no truncated files if a job times out


def combine():
    files = sorted(glob.glob(os.path.join(gt_dir, "test_[0-9][0-9][0-9][0-9].npz")))
    if files:
        data = [np.load(f) for f in files]
        comb = {k: np.stack([d[k] for d in data]) for k in data[0].files}
        save_npz(os.path.join(run_dir, "ground_truth_all.npz"), **comb)
        print(f"Combined {len(files)} ground-truth files into ground_truth_all.npz")
    else:
        print("No ground-truth files to combine.")
    parts = sorted(glob.glob(os.path.join(parts_dir, "part_[0-9][0-9][0-9].npz")))
    if parts:
        imgs = np.concatenate([np.load(f)["images"] for f in parts], axis=0)
        save_npz(os.path.join(run_dir, "unconditional_samples.npz"), images=imgs)
        print(f"Combined {len(parts)} unconditional parts ({len(imgs)} samples) into unconditional_samples.npz")
    else:
        print("No unconditional parts to combine.")


if os.environ.get("COMBINE_ONLY"):
    combine()
    raise SystemExit(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}   task {task_id}/{n_tasks}   checkpoint: {CKPT_NAME}", flush=True)

# ── Data ──────────────────────────────────────────────────────────────────────
H, W = IMAGE_SIZE
with h5py.File(data_path, "r") as f:
    ds = f["images"]
    assert tuple(ds.shape[1:]) == (H, W), f"Expected {(H, W)}, got {ds.shape[1:]}"
    n_train = ds.shape[0] - N_TEST_SET
    train = np.empty((n_train, H, W), dtype=np.uint8)
    for s in range(0, n_train, 5000):
        e = min(s + 5000, n_train)
        train[s:e] = (ds[s:e] > 0.5)
    test = ds[n_train:].astype(np.float32)
train_flat = train.reshape(n_train, -1)
print(f"Train images: {n_train}, test images: {len(test)}", flush=True)

# ── Model ─────────────────────────────────────────────────────────────────────
vocab_ckpt = torch.load(os.path.join(checkpoint_dir, "vocab_checkpoint.pt"), map_location="cpu")
vocab      = vocab_ckpt["vocab"]
mask_token = nn.Parameter(vocab_ckpt["mask_token"], requires_grad=True)

ckpt_files = sorted(
    [f for f in os.listdir(checkpoint_dir) if f.startswith("model_epoch_") and f.endswith(".pth")],
    key=lambda f: int(f.split("_")[-1].replace(".pth", ""))
)
if not ckpt_files:
    raise FileNotFoundError(f"No model_epoch_*.pth found in {checkpoint_dir}")
latest_ckpt = os.path.join(checkpoint_dir, ckpt_files[-1])
print(f"Loading checkpoint: {latest_ckpt}", flush=True)
state = torch.load(latest_ckpt, map_location="cpu")

grid_h      = H // PATCH_SIZE
grid_w      = W // PATCH_SIZE
num_patches = grid_h * grid_w
patch_dim   = PATCH_SIZE * PATCH_SIZE
emb_dim     = state["patch_proj.weight"].shape[0]
num_layers  = sum(1 for k in state if k.startswith("layers.") and k.endswith(".norm1.weight"))
print(f"Model: emb_dim={emb_dim}, layers={num_layers}, heads={NUM_HEADS}, ffn_dim={FFN_DIM}", flush=True)

model = StackedContextViT(
    vocab        = vocab,
    mask_token   = mask_token,
    patch_dim    = patch_dim,
    num_patches  = num_patches,
    grid_size    = (grid_h, grid_w),
    emb_dim      = emb_dim,
    num_heads    = NUM_HEADS,
    num_layers   = num_layers,
    ffn_dim      = FFN_DIM,
    use_pos_emb  = True,
    use_rel_bias = True,
)
missing, unexpected = model.load_state_dict(state, strict=False)
if missing:
    print(f"WARNING, missing keys (using init values): {missing}", flush=True)
if unexpected:
    print(f"WARNING, unexpected keys (ignored): {unexpected}", flush=True)
model.to(device)
model.eval()
print("Model loaded.", flush=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def sample_conditional(n, cond_idx, cond_val, order):
    """n conditional samples. Returns images float32 [n,H,W], log q(x|y) [n], log p(x) [n]."""
    all_imgs, all_lc, all_lu = [], [], []
    for _ in range(-(-n // BATCH_SIZE)):
        imgs_b, lc, lu = generate_images_batch(
            model             = model,
            patch_size        = PATCH_SIZE,
            image_size        = IMAGE_SIZE,
            batch_size        = BATCH_SIZE,
            device            = device,
            condition_indices = cond_idx,
            condition_values  = cond_val,
            generation_order  = order,
            temperature       = 1.0,
        )
        all_imgs.append(imgs_b.cpu().numpy())
        all_lc.append(lc.cpu().numpy())
        all_lu.append(lu.cpu().numpy())
    return (np.concatenate(all_imgs)[:n].astype(np.float32),
            np.concatenate(all_lc)[:n].astype(np.float64),
            np.concatenate(all_lu)[:n].astype(np.float64))


_uncond_kwargs = None


def sample_unconditional_batch():
    """One batch of unconditional samples. The exact way to call the sampler without observations is not
    known to this script, so a few variants are tried and the first that works is used (and printed)."""
    global _uncond_kwargs
    base = dict(model=model, patch_size=PATCH_SIZE, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                device=device, temperature=1.0)
    attempts = []
    if UNCOND_ORDER:
        attempts.append(dict(condition_indices=None, condition_values=None, generation_order=UNCOND_ORDER))
    attempts += [dict(),
                 dict(condition_indices=None, condition_values=None),
                 dict(condition_indices=None, condition_values=None, generation_order="raster")]
    if _uncond_kwargs is not None:
        attempts = [_uncond_kwargs]
    last = None
    for kw in attempts:
        try:
            out = generate_images_batch(**base, **kw)
            if _uncond_kwargs is None:
                print(f"Unconditional sampling works with generate_images_batch(..., {kw})", flush=True)
            _uncond_kwargs = kw
            imgs = out[0] if isinstance(out, (tuple, list)) else out
            return imgs.cpu().numpy()
        except Exception as e:          # noqa: BLE001
            last = e
    raise RuntimeError(f"Could not sample unconditionally ({type(last).__name__}: {last}). Set UNCOND_ORDER, "
                       f"or generate unconditional samples with your own script and save them as "
                       f"{os.path.join(run_dir, 'unconditional_samples.npz')} (key 'images').")


def weights_from(lc, lu):
    log_w = lu - lc
    log_w = log_w - log_w.max()
    w = np.exp(log_w)
    w /= w.sum()
    return w, float(1.0 / (w ** 2).sum())


# ══ Stage 1: two-well posterior sampling example ═════════════════════════════
def stage_example():
    path = os.path.join(run_dir, "example_two_wells.npz")
    if os.path.exists(path):
        print("[example] exists, skipping", flush=True)
        return
    t0 = time.time()
    image = test[EXAMPLE_TEST_IDX]
    idx, val = [], []
    for col in EXAMPLE_COLS:
        for row in range(H):
            idx.append(row * W + col)
            val.append(int(image[row, col] > 0.5))
    idx, val = np.array(idx, dtype=np.int64), np.array(val, dtype=np.int64)
    print(f"[example] test image {EXAMPLE_TEST_IDX}, columns {EXAMPLE_COLS} observed ({len(idx)} pixels), "
          f"{N_EXAMPLE_SAMPLES} samples", flush=True)
    torch.manual_seed(SEED)
    samples, lc, lu = sample_conditional(N_EXAMPLE_SAMPLES, idx, val, GEN_ORDER)
    w, ess = weights_from(lc, lu)
    unw = samples.mean(axis=0)
    isw = (w[:, None, None] * samples).sum(axis=0)
    print(f"[example] ESS {ess:.0f}/{N_EXAMPLE_SAMPLES}   ({time.time() - t0:.0f} s)", flush=True)
    save_npz(path,
             test_image=image.astype(np.float32), test_idx=EXAMPLE_TEST_IDX,
             cond_cols=np.array(EXAMPLE_COLS), cond_indices=idx, cond_values=val,
             samples=(samples > 0.5).astype(np.uint8),
             log_cond=lc, log_uncond=lu, weights=w, ess=ess,
             vit_mean_unw=unw.astype(np.float32), vit_mean_is=isw.astype(np.float32))


# ══ Stage 2: depth scan ══════════════════════════════════════════════════════
def pack_columns(cols):
    """[n, H] uint8 (0/1) -> uint64 [n]; row 0 is the most significant bit, so a prefix of K rows is key >> (64-K)."""
    n, h = cols.shape
    assert h <= 64
    padded = np.zeros((n, 64), dtype=np.uint8)
    padded[:, :h] = cols
    return np.packbits(padded, axis=1).view(">u8").reshape(n).astype(np.uint64)


def stage_depth_scan():
    tr_keys = pack_columns(train[:, :, WELL_COL])
    te_keys = pack_columns((test[:, :, WELL_COL] > 0.5).astype(np.uint8))
    depths = np.arange(1, H + 1)
    n_match = np.zeros((len(test), H), dtype=np.int32)
    for j, k in enumerate(depths):
        sh = np.uint64(64 - k)
        uniq, cnt = np.unique(tr_keys >> sh, return_counts=True)
        q = te_keys >> sh
        pos = np.minimum(np.searchsorted(uniq, q), len(uniq) - 1)
        n_match[:, j] = np.where(uniq[pos] == q, cnt[pos], 0)
    frac = (n_match >= MIN_MATCHES).mean(axis=0)
    ok = np.flatnonzero(frac >= PASS_FRAC)
    auto_depth = int(depths[ok.max()]) if len(ok) else 1
    depth = int(DEPTH_OVERRIDE) if DEPTH_OVERRIDE else auto_depth

    if task_id == 0:
        print(f"[depth scan] well in column {WELL_COL}, from the surface; share of the {len(test)} test images "
              f"with >= {MIN_MATCHES} matching training images (need >= {PASS_FRAC:.0%})", flush=True)
        print(f"{'depth':>6} {'share>=50':>10} {'share>=100':>11} {'share>=200':>11} {'median n':>9}")
        for k in sorted(set(list(range(4, H + 1, 4)) + [auto_depth, depth])):
            j = k - 1
            print(f"{k:>6} {(n_match[:, j] >= 50).mean():>10.2f} {(n_match[:, j] >= 100).mean():>11.2f} "
                  f"{(n_match[:, j] >= 200).mean():>11.2f} {int(np.median(n_match[:, j])):>9}", flush=True)
        print(f"[depth scan] chosen depth: {depth}" + ("  (override)" if DEPTH_OVERRIDE else ""), flush=True)
        path = os.path.join(run_dir, "depth_scan.npz")
        if not os.path.exists(path):
            save_npz(path, depths=depths, n_match=n_match, well_col=WELL_COL, min_matches=MIN_MATCHES,
                     pass_frac=PASS_FRAC, chosen_depth=depth, auto_depth=auto_depth)
    return depth, n_match[:, depth - 1]


# ══ Stage 3: ground-truth test with one deep well ════════════════════════════
def stage_ground_truth(depth, n_at_depth):
    passing = np.flatnonzero(n_at_depth >= MIN_MATCHES)
    selected = passing[:N_TESTS]
    print(f"[ground truth] depth {depth}: {len(passing)} test images qualify, using the first {len(selected)}", flush=True)
    if len(selected) < N_TESTS:
        print(f"WARNING: only {len(selected)} of the requested {N_TESTS} test images qualify", flush=True)
    idx = np.arange(depth) * W + WELL_COL
    mine = list(range(task_id, len(selected), n_tasks))
    t_start = time.time()
    done = 0
    for slot in mine:
        out_path = os.path.join(gt_dir, f"test_{slot:04d}.npz")
        if os.path.exists(out_path):
            print(f"[{slot}] exists, skipping", flush=True)
            continue
        img_idx = int(selected[slot])
        image = test[img_idx]
        vals = (image.reshape(-1)[idx] > 0.5).astype(np.int64)
        mask = (train_flat[:, idx] == vals).all(axis=1)
        n_match = int(mask.sum())
        ref_mean = train[mask].mean(axis=0).astype(np.float32)
        ref_var_floor = float((ref_mean * (1 - ref_mean)).mean() / n_match)

        torch.manual_seed(SEED + slot)
        samples, lc, lu = sample_conditional(N_SAMPLES, idx, vals, GEN_ORDER)
        w, ess = weights_from(lc, lu)
        unw = samples.mean(axis=0)
        isw = (w[:, None, None] * samples).sum(axis=0)
        std = np.sqrt(np.maximum((w[:, None, None] * samples ** 2).sum(axis=0) - isw ** 2, 0))

        mse_unw = ((unw - ref_mean) ** 2).mean()
        mse_is = ((isw - ref_mean) ** 2).mean()
        done += 1
        el = time.time() - t_start
        print(f"[{slot}] test image {img_idx}: {n_match} matches; MSE vs ref unw={mse_unw:.5f} IS={mse_is:.5f} "
              f"(ref noise ~{ref_var_floor:.5f}); ESS={ess:.0f}/{N_SAMPLES}; "
              f"{el / done:.0f} s per test, {len(mine)} tests for this task", flush=True)
        save_npz(out_path,
                 ref_mean=ref_mean, n_match=n_match, ref_var_floor=ref_var_floor,
                 vit_mean_unw=unw.astype(np.float32), vit_mean_is=isw.astype(np.float32),
                 vit_std_is=std.astype(np.float32), ess=ess,
                 cond_indices=idx, cond_values=vals, test_idx=img_idx,
                 test_image=image.astype(np.float32))


# ══ Stage 4: unconditional samples (ViT prior mean) ══════════════════════════
def stage_unconditional():
    path = os.path.join(parts_dir, f"part_{task_id:03d}.npz")
    if os.path.exists(path):
        print("[unconditional] part exists, skipping", flush=True)
        return
    n_batches = -(-N_UNCOND // n_tasks // BATCH_SIZE)
    torch.manual_seed(SEED + 10_000 + task_id)
    t0 = time.time()
    out = []
    for b in range(n_batches):
        out.append((sample_unconditional_batch() > 0.5).astype(np.uint8))
        print(f"[unconditional] batch {b + 1}/{n_batches}  ({time.time() - t0:.0f} s)", flush=True)
    save_npz(path, images=np.concatenate(out, axis=0))


# ── Run ───────────────────────────────────────────────────────────────────────
if task_id == 0:
    stage_example()
depth, n_at_depth = stage_depth_scan()
stage_ground_truth(depth, n_at_depth)
stage_unconditional()

if n_tasks == 1:
    combine()
print("Done.", flush=True)

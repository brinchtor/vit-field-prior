"""
attention_maps.py  (TORCH ONLY -- no matplotlib)

For each trained ViT prior (models A-D), run the model's own unconditional
generation (raster order, mask_rate=0, exactly as in generate_image) up to the
SAME step of the recursion, then read the log-attention map
QK^T/sqrt(dh) + R from the current query patch to all key patches.

The conditioning context therefore differs between models, because each model
generates its own content, but the query position and the recursion step are
identical, so the attention maps are comparable.

Saved arrays:
    labels        [M]            model labels
    attn_maps     [M, Gh, Gw]    head-mean QK^T/sqrt(dh)+R, query row reshaped
    context_imgs  [M, H, W]      each model's partial generation; 0.5 = masked
    query_patch   [2]            (row, col) of the query patch on the patch grid
    grid          [2]            (Gh, Gw)
    image_size    [2]            (H, W)
    patch_size    scalar
    meta_layer, meta_step
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ViT_rectangle import StackedContextViT

# ── Config ──────────────────────────────────────────────────────────────────
PATCH_SIZE   = 2
IMAGE_SIZE   = (12, 24)          # (H, W)
NUM_HEADS    = 2
FFN_DIM      = 256
LAYER        = 0                # transformer layer whose attention is shown
STEP_FRAC    = ((12*3.5)/(12*6))            # fraction of patches generated before the query step
SEED         = 0

MODELS = {
    "A: stationary Ising":        "outputs-model-A",
    "B: non-stationary Ising":    "outputs-model-B",
    "C: isotropic 2nd-order":     "outputs-model-C",
    "D: anisotropic 2nd-order":   "outputs-model-D",
}
OUT_NPZ = "attention_data.npz"


def load_model(checkpoint_dir, device):
    vocab_ckpt = torch.load(os.path.join(checkpoint_dir, "vocab_checkpoint.pt"),
                            map_location="cpu")
    vocab      = vocab_ckpt["vocab"]
    mask_token = nn.Parameter(vocab_ckpt["mask_token"], requires_grad=True)

    ckpt_files = sorted(
        [f for f in os.listdir(checkpoint_dir)
         if f.startswith("model_epoch_") and f.endswith(".pth")],
        key=lambda f: int(f.split("_")[-1].replace(".pth", "")),
    )
    if not ckpt_files:
        raise FileNotFoundError(f"No model_epoch_*.pth in {checkpoint_dir}")
    state = torch.load(os.path.join(checkpoint_dir, ckpt_files[-1]), map_location="cpu")

    H, W        = IMAGE_SIZE
    grid_h      = H // PATCH_SIZE
    grid_w      = W // PATCH_SIZE
    num_patches = grid_h * grid_w
    patch_dim   = PATCH_SIZE * PATCH_SIZE
    emb_dim     = state["patch_proj.weight"].shape[0]
    num_layers  = sum(1 for k in state
                      if k.startswith("layers.") and k.endswith(".norm1.weight"))

    model = StackedContextViT(
        vocab=vocab, mask_token=mask_token, patch_dim=patch_dim,
        num_patches=num_patches, grid_size=(grid_h, grid_w), emb_dim=emb_dim,
        num_heads=NUM_HEADS, num_layers=num_layers, ffn_dim=FFN_DIM,
        use_pos_emb=True, use_rel_bias=True,
    )
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def generate_partial(model, t, order):
    """Run unconditional generation (raster order, mask_rate=0) for the first t
    patches of `order`. Returns the partial patch tensor [N, P] with order[:t]
    sampled and the rest still masked, plus the set of filled patch indices."""
    device = next(model.parameters()).device
    N = model.grid_h * model.grid_w
    P = PATCH_SIZE * PATCH_SIZE
    vocab = model.vocab.to(device)
    gp = model.mask_token.detach().to(device).unsqueeze(0).expand(N, P).clone()
    for step in range(t):
        p_idx = order[step]
        logits, _ = model(gp.unsqueeze(0), 0.0)          # [1, N, V]
        probs = F.softmax(logits[0, p_idx], dim=-1)
        s = torch.multinomial(probs, num_samples=1).item()
        gp[p_idx] = vocab[s]
    return gp, set(order[:t])


@torch.no_grad()
def attn_map_for_query(model, context_patches, q):
    """Head-mean QK^T/sqrt(dh)+R from query q to all keys at LAYER -> [Gh, Gw]."""
    device = next(model.parameters()).device
    Gh, Gw = model.grid_h, model.grid_w
    N = Gh * Gw
    emb = model.patch_proj.out_features
    dh  = emb // NUM_HEADS

    idx  = torch.arange(N, device=device)
    dr = (idx // Gw)[:, None] - (idx // Gw)[None, :]
    dc = (idx %  Gw)[:, None] - (idx %  Gw)[None, :]
    R = model.rel_bias_table[dr + (Gh - 1), dc + (Gw - 1)].detach()

    x = context_patches.to(device).unsqueeze(0)
    x = model.patch_proj(x)
    if model.use_pos_emb:
        x = x + model.pos_emb.unsqueeze(0)
    for l in range(LAYER):
        x = model.layers[l](x, attn_mask=R)

    attn = model.layers[LAYER].attn
    Wq, Wk, _ = attn.in_proj_weight.chunk(3, 0)
    bq, bk, _ = attn.in_proj_bias.chunk(3, 0)
    qv = (x @ Wq.T + bq).view(1, N, NUM_HEADS, dh).permute(0, 2, 1, 3)
    kv = (x @ Wk.T + bk).view(1, N, NUM_HEADS, dh).permute(0, 2, 1, 3)
    scores = (qv @ kv.transpose(-1, -2)) / math.sqrt(dh) + R[None, None]
    row = scores[0].mean(0)[q]
    return row.view(Gh, Gw).cpu().numpy().astype(np.float32)


def context_display(gp, filled, grid, image_size, patch_size):
    """Partial generation -> display image [H,W]: filled patches 0/1, masked 0.5."""
    Gh, Gw = grid
    H, W = image_size
    ps = patch_size
    disp = np.full((H, W), 0.5, dtype=np.float32)
    gp_np = gp.detach().cpu().numpy()
    for p in filled:
        pr, pc = divmod(p, Gw)
        disp[pr*ps:(pr+1)*ps, pc*ps:(pc+1)*ps] = gp_np[p].reshape(ps, ps)
    return disp


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    labels = list(MODELS.keys())
    ref = load_model(MODELS[labels[0]], device)
    Gh, Gw = ref.grid_h, ref.grid_w
    N = Gh * Gw

    order = list(range(N))                       # raster generation order
    t = int(STEP_FRAC * N)
    q = order[t]
    qr, qc = divmod(q, Gw)

    maps, contexts = [], []
    for i, name in enumerate(labels):
        torch.manual_seed(SEED + i)              # reproducible, independent per model
        model = ref if name == labels[0] else load_model(MODELS[name], device)
        gp, filled = generate_partial(model, t, order)
        maps.append(attn_map_for_query(model, gp, q))
        contexts.append(context_display(gp, filled, (Gh, Gw), IMAGE_SIZE, PATCH_SIZE))
        print(f"generated + attn computed: {name}")

    np.savez_compressed(
        OUT_NPZ,
        labels=np.array(labels),
        attn_maps=np.stack(maps),
        context_imgs=np.stack(contexts),
        query_patch=np.array((qr, qc), dtype=np.int64),
        grid=np.array((Gh, Gw), dtype=np.int64),
        image_size=np.array(IMAGE_SIZE, dtype=np.int64),
        patch_size=np.int64(PATCH_SIZE),
        meta_layer=np.int64(LAYER),
        meta_step=np.int64(t),
    )
    print(f"Saved {OUT_NPZ}")
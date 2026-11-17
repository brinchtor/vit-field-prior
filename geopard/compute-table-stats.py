import torch
import numpy as np
import h5py
import os


def compute_table_stats(vocab: torch.Tensor, train_patches: torch.Tensor, patch_size: int):
    """
    Compute the stats needed for the vocabulary-capacity table:
      - total possible patches (2^(patch_size^2))
      - vocab size (|V|)
      - occurrence coverage (% of training patch instances whose pattern is in vocab)
      - per-column pattern availability (min/max present out of 2^patch_size, across columns)
    """
    V = vocab.shape[0]
    total_possible = 2 ** (patch_size * patch_size)

    patch_powers = (2 ** torch.arange(patch_size * patch_size)).to(vocab.device)
    vocab_ints = (vocab * patch_powers).sum(dim=1).long()
    vocab_set = set(vocab_ints.cpu().numpy().tolist())

    train_ints = (train_patches * patch_powers).sum(dim=1).long()
    train_ints_np = train_ints.cpu().numpy()
    in_vocab_mask = np.isin(train_ints_np, list(vocab_set))
    occurrence_coverage = in_vocab_mask.mean() * 100

    distinct_train_patterns = len(set(train_ints_np.tolist()))
    duplication_ratio = len(train_ints_np) / distinct_train_patterns

    n_col_required = 2 ** patch_size
    col_present_counts = []
    for col in range(patch_size):
        col_indices = [row * patch_size + col for row in range(patch_size)]
        col_patterns = vocab[:, col_indices]
        col_powers = (2 ** torch.arange(patch_size)).to(vocab.device)
        col_ints = (col_patterns * col_powers).sum(dim=1).long()
        n_present = len(set(col_ints.cpu().numpy().tolist()))
        col_present_counts.append(n_present)

    return {
        "patch_size": patch_size,
        "total_possible_patches": total_possible,
        "vocab_size": V,
        "distinct_train_patterns": distinct_train_patterns,
        "duplication_ratio": duplication_ratio,
        "occurrence_coverage_pct": occurrence_coverage,
        "col_required": n_col_required,
        "col_present_min": min(col_present_counts),
        "col_present_max": max(col_present_counts),
        "col_complete": min(col_present_counts) == n_col_required,
    }


def load_images_h5(path: str, dataset_key: str = "images") -> torch.Tensor:
    with h5py.File(path, "r") as f:
        arr = f[dataset_key][:]  # [N, H, W]
    return torch.from_numpy(arr).float()


def extract_all_patches(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    images: [N, H, W] binary tensor.
    Returns: [N * n_patches_per_image, patch_size^2] tensor of every
    non-overlapping patch occurrence across the whole set.
    """
    N, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0, f"H={H}, W={W} not divisible by patch_size={patch_size}"
    patches = images.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # -> [N, H//p, W//p, p, p]
    patches = patches.reshape(-1, patch_size * patch_size)
    return patches


def print_markdown_table(rows: list[dict]):
    print("| Resolution | Patch size | Possible patches | Vocab size | Occurrence coverage | Distinct patterns (dup. ratio) | Per-column patterns needed | Per-column patterns available |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        avail = (
            f"{r['col_required']} (all)" if r["col_complete"]
            else f"{r['col_present_min']}–{r['col_present_max']}"
        )
        print(
            f"| {r.get('resolution', '?')} | {r['patch_size']}×{r['patch_size']} "
            f"| {r['total_possible_patches']:,} | {r['vocab_size']:,} "
            f"| {r['occurrence_coverage_pct']:.2f}% "
            f"| {r['distinct_train_patterns']:,} ({r['duplication_ratio']:.0f}×) "
            f"| {r['col_required']} | {avail} |"
        )


# ── Usage ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    configs = [
        {
            "resolution": "64×64",
            "patch_size": 4,
            "vocab_path": "checkpoints_fluvial_64/vocab_checkpoint.pt",
            "data_path": "data/fluvial_binary_64x64.h5",
        },
        {
            "resolution": "128×128",
            "patch_size": 8,
            "vocab_path": "checkpoints_fluvial/vocab_checkpoint.pt",  # assumed — confirm this path
            "data_path": "data/fluvial_binary_128x128.h5",
        },
    ]

    rows = []
    for cfg in configs:
        vocab_ckpt = torch.load(cfg["vocab_path"], map_location="cpu")
        vocab = vocab_ckpt["vocab"]  # [V, patch_dim]

        images = load_images_h5(cfg["data_path"])
        train_patches = extract_all_patches(images, cfg["patch_size"])

        stats = compute_table_stats(vocab, train_patches, cfg["patch_size"])
        stats["resolution"] = cfg["resolution"]
        rows.append(stats)

    print_markdown_table(rows)
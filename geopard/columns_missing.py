import torch
import numpy as np

def check_column_coverage(vocab: torch.Tensor, patch_size: int = 8):
    """
    Check which column positions in the vocab have complete coverage,
    i.e. all 2^patch_size binary patterns present.

    Args:
        vocab: [V, patch_size^2] binary tensor of vocab entries
        patch_size: patch height/width (e.g. 8 for 8x8 patches)

    Returns:
        coverage: dict mapping column_idx -> (n_present, n_missing, missing_patterns)
    """
    V = vocab.shape[0]
    n_required = 2 ** patch_size

    coverage = {}

    for col in range(patch_size):
        # Extract the column pixels from each vocab entry
        # For an 8x8 patch flattened row-major, column col has indices:
        # col, col+patch_size, col+2*patch_size, ..., col+7*patch_size
        col_indices = [row * patch_size + col for row in range(patch_size)]
        col_patterns = vocab[:, col_indices]  # [V, patch_size]

        # Convert each pattern to an integer for easy comparison
        # Pattern [b0, b1, ..., b7] -> integer
        powers = (2 ** torch.arange(patch_size)).to(vocab.device)
        pattern_ints = (col_patterns * powers).sum(dim=1).long()  # [V]

        present = set(pattern_ints.cpu().numpy().tolist())
        all_patterns = set(range(n_required))
        missing = all_patterns - present

        coverage[col] = {
            "n_present": len(present),
            "n_missing": len(missing),
            "complete": len(missing) == 0,
            "missing_patterns": sorted(missing),
        }

    return coverage


def print_coverage_report(coverage: dict, patch_size: int = 8):
    n_required = 2 ** patch_size
    print(f"Column coverage report (patch_size={patch_size}, {n_required} patterns required per column)")
    print("-" * 60)
    all_complete = True
    for col, info in coverage.items():
        status = "COMPLETE" if info["complete"] else f"MISSING {info['n_missing']}"
        print(f"  Column {col}: {info['n_present']}/{n_required} patterns present  [{status}]")
        if not info["complete"]:
            all_complete = False
    print("-" * 60)
    if all_complete:
        print("All columns have complete coverage!")
    else:
        print("Some columns are incomplete — well conditioning may fall back to unconstrained sampling.")


# ── Usage ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import torch.nn as nn

    checkpoint_dir = "checkpoints_fluvial"
    vocab_ckpt = torch.load(os.path.join(checkpoint_dir, "vocab_checkpoint.pt"), map_location="cpu")
    vocab = vocab_ckpt["vocab"]  # [V, patch_dim]

    patch_size = 8  # change to 4 for 4x4 patches
    print(f"Vocab size: {vocab.shape[0]}, patch_dim: {vocab.shape[1]}")

    coverage = check_column_coverage(vocab, patch_size=patch_size)
    print_coverage_report(coverage, patch_size=patch_size)
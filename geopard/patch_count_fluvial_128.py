import h5py
import numpy as np
from collections import Counter
import time

def count_unique_patches(data_path, patch_size=8):
    print("Loading data...")
    t0 = time.time()
    with h5py.File(data_path, "r") as f:
        imgs = f["images"][:].astype(np.uint8)
    print(f"Loaded in {time.time()-t0:.1f}s — shape: {imgs.shape}")

    N, H, W = imgs.shape
    grid_h = H // patch_size
    grid_w = W // patch_size
    total_patches = N * grid_h * grid_w

    print(f"\nDataset:          {N:,} images")
    print(f"Image size:       {H}x{W}")
    print(f"Patch size:       {patch_size}x{patch_size}")
    print(f"Patches/image:    {grid_h * grid_w} ({grid_h}x{grid_w} grid)")
    print(f"Total patches:    {total_patches:,}")
    print(f"\nExtracting patches...")

    t1 = time.time()
    # Reshape into patches: [N, grid_h, patch_size, grid_w, patch_size]
    patches = imgs.reshape(N, grid_h, patch_size, grid_w, patch_size)
    # Reorder to [N, grid_h, grid_w, patch_size, patch_size]
    patches = patches.transpose(0, 1, 3, 2, 4)
    # Flatten to [N*grid_h*grid_w, patch_size*patch_size]
    patches = np.ascontiguousarray(
        patches.reshape(-1, patch_size * patch_size)
    )
    print(f"Extracted in {time.time()-t1:.1f}s")

    print("Counting unique patches...")
    t2 = time.time()
    # View each row as a single void element for fast hashing
    patches_void = patches.view(
        np.dtype((np.void, patches.dtype.itemsize * patches.shape[1]))
    ).reshape(-1)
    unique_void, counts = np.unique(patches_void, return_counts=True)
    print(f"Counted in {time.time()-t2:.1f}s")

    n_unique = len(unique_void)
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Total patches:          {total_patches:,}")
    print(f"Unique patches:         {n_unique:,}")
    print(f"Duplication ratio:      {total_patches/n_unique:.1f}x")
    print(f"\nFrequency breakdown:")
    print(f"  Singletons (==1):     {(counts==1).sum():,}")
    print(f"  Seen < 5 times:       {(counts<5).sum():,}")
    print(f"  Seen < 10 times:      {(counts<10).sum():,}")
    print(f"  Seen < 100 times:     {(counts<100).sum():,}")
    print(f"  Seen >= 50 times:    {(counts>=50).sum():,}")
    print(f"  Seen >= 100 times:    {(counts>=100).sum():,}")
    print(f"  Seen >= 1000 times:   {(counts>=1000).sum():,}")
    print(f"\nMost common patch:      {counts.max():,} times")
    print(f"Median frequency:       {np.median(counts):.0f} times")

    # How many patches needed to cover X% of all patch occurrences
    counts_sorted = np.sort(counts)[::-1]
    cumsum = np.cumsum(counts_sorted)
    for pct in [50, 90, 95, 99]:
        n_needed = np.searchsorted(cumsum, total_patches * pct / 100) + 1
        print(f"  Top {n_needed:,} patches cover {pct}% of all occurrences")
        
        # How well does your actual vocab size cover occurrences?
    for vocab_size in [6000, 4096, 2048, 1024]:
        coverage_pct = cumsum[min(vocab_size, len(cumsum)) - 1] / total_patches * 100
        print(f"  Top {vocab_size:,} patches cover {coverage_pct:.2f}% of all occurrences")
    
    print(f"\nTotal time: {time.time()-t0:.1f}s")

    return unique_void, counts


if __name__ == "__main__":
    data_path = "data/fluvial_binary_128x128.h5"  # adjust path
    unique, counts = count_unique_patches(data_path, patch_size=8)
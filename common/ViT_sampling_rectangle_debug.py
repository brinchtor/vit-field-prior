import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from ViT_rectangle import *

import torch
import torch.nn.functional as F
import math
import numpy as np

def _parse_image_shape(image_size):
    """Accept either an int (square) or a (H, W) tuple/list."""
    if isinstance(image_size, (tuple, list)):
        if len(image_size) != 2:
            raise ValueError(f"image_size as tuple/list must be (H, W), got {image_size}")
        H, W = int(image_size[0]), int(image_size[1])
    else:
        H = W = int(image_size)
    return H, W

def _grid_from_shape(H, W, patch_size):
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"H,W must be divisible by patch_size. Got H={H}, W={W}, patch_size={patch_size}")
    Gh = H // patch_size
    Gw = W // patch_size
    N = Gh * Gw
    return Gh, Gw, N

def log_likelihood_evaluation_batch(
    model: nn.Module,
    batch_images: torch.Tensor,         # [B, H, W]
    patch_size: int,
    condition_indices: np.ndarray = None,
    condition_values: np.ndarray = None,
    generation_order: str = "manhattan"
) -> torch.Tensor:
    """
    For each image in the batch, compute the log-likelihood under the ViT
    by “revealing” patches one by one (in the chosen order) and accumulating
    log p(patch_i | previously revealed patches).

    Notes:
      * Supports rectangular images.
      * condition_indices are flattened with row-major indexing using width=W.

    Returns:
        log_likes: Tensor of shape [B], total log-likelihood per image.
    """
    model.eval()
    device = next(model.parameters()).device

    B, H, W = batch_images.shape
    Gh, Gw, N = _grid_from_shape(H, W, patch_size)
    P = patch_size * patch_size

    # 1) Extract all true patches from batch_images
    patches = BinaryImageDataset.batch_to_patches(batch_images, patch_size).to(device)  # [B, N, P]

    # 2) Start with mask-token everywhere
    mask_tok = model.mask_token.detach().to(device)                          # [P]
    gen = mask_tok.unsqueeze(0).unsqueeze(0).expand(B, N, P).clone()         # [B,N,P]

    # 3) Prepare conditioning constraints (assumes same indices across batch; values can be 1D or [B,k])
    cond_by_patch = {}  # patch_idx -> list[(local_idx, val)]
    observed = []
    if condition_indices is not None and condition_values is not None:
        condition_indices = np.array(condition_indices)
        condition_values = np.array(condition_values)
        if condition_values.ndim == 2:
            # Use first row for patch-local structure; values per-image not supported here without more logic.
            condition_values_use = condition_values[0]
        else:
            condition_values_use = condition_values

        for pix, val in zip(condition_indices, condition_values_use):
            r, c = divmod(int(pix), W)
            pr, pc = r // patch_size, c // patch_size
            patch_idx = pr * Gw + pc
            lr, lc = r % patch_size, c % patch_size
            local = lr * patch_size + lc
            cond_by_patch.setdefault(patch_idx, []).append((local, int(val)))
        observed = sorted(cond_by_patch.keys())

    # 4) Build sampling order
    all_idxs = set(range(N))
    unobs = list(all_idxs - set(observed))

    if generation_order == "manhattan":
        if observed:
            dist_list = []
            obs_rc = [divmod(o, Gw) for o in observed]
            for p in unobs:
                pr, pc = divmod(p, Gw)
                d = 0
                for orow, ocol in obs_rc:
                    d += abs(pr - orow) + abs(pc - ocol)
                dist_list.append((p, d))
            dist_list.sort(key=lambda x: x[1])
            unobs = [p for p, _ in dist_list]
        else:
            unobs.sort()
    elif generation_order == "raster":
        unobs.sort()
    elif generation_order == "random":
        np.random.shuffle(unobs)
    else:
        raise ValueError(f"Unknown generation_order: {generation_order}")

    order = observed + unobs

    # 5) Iterate and accumulate
    log_likes = torch.zeros(B, device=device)
    with torch.no_grad():
        for p_idx in order:
            logits, _ = model(gen, mask_rate=0.0)        # [B,N,V]
            logits_p = logits[:, p_idx, :].clone()       # [B, V]

            # If p_idx is conditioned, mask out invalid vocab entries
            if p_idx in cond_by_patch:
                valid = torch.ones(model.vocab_size, dtype=torch.bool, device=device)
                for local, val in cond_by_patch[p_idx]:
                    valid &= (model.vocab[:, local] == val)
                logits_p[:, ~valid] = -1e9

            # Compute true-patch vocab index by NN in vocab
            true_patch = patches[:, p_idx, :]            # [B,P]
            dmat = torch.cdist(true_patch, model.vocab.to(device))  # [B, V]
            true_idx = dmat.argmin(dim=1)                # [B]

            logp = F.log_softmax(logits_p, dim=-1)       # [B, V]
            log_likes += logp[torch.arange(B, device=device), true_idx]

            # Reveal true patch
            gen[:, p_idx, :] = patches[:, p_idx, :]

    return log_likes

def log_likelihood_evaluation_batch_debug(
    model: nn.Module,
    batch_images: torch.Tensor,         # [B, H, W]
    patch_size: int,
    condition_indices: np.ndarray = None,
    condition_values: np.ndarray = None,
    generation_order: str = "manhattan",
    return_cpu: bool = False,           # handy for histogramming
):
    """
    Debug version: returns every per-step loglik contribution (the term added to log_likes).

    Returns:
        log_likes: Tensor [B]
        contribs:  Tensor [B, T]  (T = len(order))
        debug: dict with:
            - "order": List[int] length T (patch indices in reveal order)
            - "true_idx": LongTensor [B, T] vocab indices chosen as ground-truth each step
            - "is_conditioned_step": BoolTensor [T] True if that step had conditioning
    """
    model.eval()
    device = next(model.parameters()).device

    B, H, W = batch_images.shape
    Gh, Gw, N = _grid_from_shape(H, W, patch_size)
    P = patch_size * patch_size

    # 1) Extract all true patches from batch_images
    patches = BinaryImageDataset.batch_to_patches(batch_images, patch_size).to(device)  # [B, N, P]

    # 2) Start with mask-token everywhere
    mask_tok = model.mask_token.detach().to(device)                          # [P]
    gen = mask_tok.unsqueeze(0).unsqueeze(0).expand(B, N, P).clone()         # [B,N,P]

    # 3) Prepare conditioning constraints
    cond_by_patch = {}  # patch_idx -> list[(local_idx, val)]
    observed = []
    if condition_indices is not None and condition_values is not None:
        condition_indices = np.array(condition_indices)
        condition_values = np.array(condition_values)

        if condition_values.ndim == 2:
            condition_values_use = condition_values[0]
        else:
            condition_values_use = condition_values

        for pix, val in zip(condition_indices, condition_values_use):
            r, c = divmod(int(pix), W)
            pr, pc = r // patch_size, c // patch_size
            patch_idx = pr * Gw + pc
            lr, lc = r % patch_size, c % patch_size
            local = lr * patch_size + lc
            cond_by_patch.setdefault(patch_idx, []).append((local, int(val)))
        observed = sorted(cond_by_patch.keys())

    # 4) Build sampling order
    all_idxs = set(range(N))
    unobs = list(all_idxs - set(observed))

    if generation_order == "manhattan":
        if observed:
            dist_list = []
            obs_rc = [divmod(o, Gw) for o in observed]
            for p in unobs:
                pr, pc = divmod(p, Gw)
                d = 0
                for orow, ocol in obs_rc:
                    d += abs(pr - orow) + abs(pc - ocol)
                dist_list.append((p, d))
            dist_list.sort(key=lambda x: x[1])
            unobs = [p for p, _ in dist_list]
        else:
            unobs.sort()
    elif generation_order == "raster":
        unobs.sort()
    elif generation_order == "random":
        np.random.shuffle(unobs)
    else:
        raise ValueError(f"Unknown generation_order: {generation_order}")

    order = observed + unobs
    T = len(order)

    # 5) Iterate and accumulate + collect contributions
    log_likes = torch.zeros(B, device=device)
    contrib_list = []          # each element: Tensor [B]
    true_idx_list = []         # each element: LongTensor [B]
    is_cond_step = torch.tensor([p in cond_by_patch for p in order], device=device, dtype=torch.bool)

    vocab = model.vocab.to(device)  # avoid repeated .to(device)

    with torch.no_grad():
        for p_idx in order:
            logits, _ = model(gen, mask_rate=0.0)        # [B,N,V]
            logits_p = logits[:, p_idx, :].clone()       # [B, V]

            # If p_idx is conditioned, mask out invalid vocab entries
            if p_idx in cond_by_patch:
                valid = torch.ones(model.vocab_size, dtype=torch.bool, device=device)
                for local, val in cond_by_patch[p_idx]:
                    valid &= (vocab[:, local] == val)
                logits_p[:, ~valid] = -1e9

            # Compute true-patch vocab index by NN in vocab
            true_patch = patches[:, p_idx, :]            # [B,P]
            dmat = torch.cdist(true_patch, vocab)        # [B, V]
            true_idx = dmat.argmin(dim=1)                # [B]

            logp = F.log_softmax(logits_p, dim=-1)       # [B, V]
            contrib = logp[torch.arange(B, device=device), true_idx]  # [B]

            # Accumulate + record
            log_likes += contrib
            contrib_list.append(contrib)
            true_idx_list.append(true_idx)

            # Reveal true patch
            gen[:, p_idx, :] = patches[:, p_idx, :]

    contribs = torch.stack(contrib_list, dim=1)      # [B, T]
    true_idx_mat = torch.stack(true_idx_list, dim=1) # [B, T]

    debug = {
        "order": order,
        "true_idx": true_idx_mat,
        "is_conditioned_step": is_cond_step,
    }

    if return_cpu:
        log_likes = log_likes.detach().cpu()
        contribs = contribs.detach().cpu()
        debug = {
            "order": debug["order"],
            "true_idx": debug["true_idx"].detach().cpu(),
            "is_conditioned_step": debug["is_conditioned_step"].detach().cpu(),
        }

    return log_likes, contribs, debug

########################################
# Generates conditional images
########################################

def generate_image(
    model,
    patch_size,
    image_size,
    condition_indices=None,
    condition_values=None,
    generation_order="manhattan"
):
    """
    Generate an image (tensor [H, W]) conditionally based on observed cells.

    image_size can be either:
      - int (square image: image_size x image_size), or
      - (H, W) tuple for rectangular images.

    condition_indices are flattened full-image indices (row-major) using width=W.
    """
    model.eval()
    device = next(model.parameters()).device

    H, W = _parse_image_shape(image_size)
    Gh, Gw, total_patches = _grid_from_shape(H, W, patch_size)
    patch_dim = patch_size * patch_size

    # Initialize all patches as mask token: [total_patches, patch_dim]
    mask_patch = model.mask_token.detach().to(device)
    generated_patches = mask_patch.unsqueeze(0).expand(total_patches, patch_dim).clone()

    # Build patch-local constraints
    conditions_by_patch = {}
    if condition_indices is not None and condition_values is not None:
        for cond_idx, cond_val in zip(condition_indices, condition_values):
            cond_idx = int(cond_idx)
            pixel_row = cond_idx // W
            pixel_col = cond_idx % W

            patch_row = pixel_row // patch_size
            patch_col = pixel_col // patch_size
            patch_idx = patch_row * Gw + patch_col

            local_row = pixel_row % patch_size
            local_col = pixel_col % patch_size
            local_idx = local_row * patch_size + local_col

            conditions_by_patch.setdefault(patch_idx, []).append((local_idx, int(cond_val)))

    observed_patch_ids = sorted(list(conditions_by_patch.keys()))

    # Build generation order for unobserved patches
    all_patch_indices = set(range(total_patches)) - set(observed_patch_ids)

    if generation_order == "manhattan":
        distance_list = []
        obs_rc = [divmod(op, Gw) for op in observed_patch_ids]
        for p in all_patch_indices:
            row, col = divmod(p, Gw)
            dist_sum = 0
            for orow, ocol in obs_rc:
                dist_sum += abs(row - orow) + abs(col - ocol)
            distance_list.append((p, dist_sum))
        distance_list.sort(key=lambda x: x[1])
        unobserved_patch_ids = [p for p, _ in distance_list]

    elif generation_order == "raster":
        unobserved_patch_ids = sorted(list(all_patch_indices))

    elif generation_order == "random":
        unobserved_patch_ids = list(all_patch_indices)
        np.random.shuffle(unobserved_patch_ids)

    else:
        raise ValueError(
            f"Unknown generation_order: {generation_order}. Must be one of ['manhattan', 'raster', 'random']."
        )

    sampling_order = observed_patch_ids + unobserved_patch_ids

    log_likelihood_sum = 0.0

    # Generate patch-by-patch (batch size 1)
    for p_idx in sampling_order:
        current_tokens = generated_patches.unsqueeze(0).clone()  # [1, total_patches, patch_dim]
        with torch.no_grad():
            logits, _ = model(current_tokens, mask_rate=0.0)      # [1, total_patches, vocab_size]
        logits_current = logits[0, p_idx].clone()                 # [vocab_size]

        if p_idx in conditions_by_patch:
            valid_candidates = torch.ones(model.vocab.shape[0], dtype=torch.bool, device=device)
            for local_idx, cond_val in conditions_by_patch[p_idx]:
                valid_candidates &= (model.vocab[:, local_idx].to(device) == cond_val)
            logits_current[~valid_candidates] = -float("inf")

        probs = F.softmax(logits_current, dim=-1)
        sampled_idx = torch.multinomial(probs, num_samples=1).item()
        log_likelihood_sum += torch.log(probs[sampled_idx] + 1e-10).item()

        generated_patches[p_idx] = model.vocab.to(device)[sampled_idx]

    generated_image = patches_to_image(generated_patches.detach().cpu(), (H, W), patch_size)
    return generated_image, log_likelihood_sum


def generate_images_batch(
    model,
    patch_size,
    image_size,
    batch_size,
    device,
    condition_indices=None,
    condition_values=None,
    generation_order="manhattan",
    temperature=1.0,
):
    """
    Generate a batch of images in parallel.

    image_size can be int (square) or (H, W).
    condition_indices/values (if given) are assumed to apply to all images.
    """
    model.eval()
    H, W = _parse_image_shape(image_size)
    Gh, Gw, total_patches = _grid_from_shape(H, W, patch_size)
    patch_dim = patch_size * patch_size

    # Initialize: [batch, total_patches, patch_dim] filled with mask token
    mask_patch = model.mask_token.detach().to(device)
    generated_patches = mask_patch.unsqueeze(0).unsqueeze(0).expand(batch_size, total_patches, patch_dim).clone()

    # Conditioning constraints by patch
    conditions_by_patch = {}
    observed_patch_ids = []
    if condition_indices is not None and condition_values is not None:
        condition_indices = np.array(condition_indices)
        condition_values = np.array(condition_values)

        if condition_values.ndim == 1:
            condition_values_batch = np.tile(condition_values, (batch_size, 1))
        else:
            condition_values_batch = condition_values

        # Build constraints (assumes same indices for all images; use first row for structure)
        for cond_idx, cond_val in zip(condition_indices, condition_values_batch[0]):
            cond_idx = int(cond_idx)
            pixel_row = cond_idx // W
            pixel_col = cond_idx % W

            patch_row = pixel_row // patch_size
            patch_col = pixel_col // patch_size
            patch_idx = patch_row * Gw + patch_col

            local_row = pixel_row % patch_size
            local_col = pixel_col % patch_size
            local_idx = local_row * patch_size + local_col
            conditions_by_patch.setdefault(patch_idx, []).append((local_idx, int(cond_val)))

        observed_patch_ids = sorted(conditions_by_patch.keys())
    else:
        condition_values_batch = None

    # Determine generation order for patches
    all_patches = set(range(total_patches))
    unobserved_patch_ids = list(all_patches - set(observed_patch_ids))

    if generation_order == "manhattan" and observed_patch_ids:
        distance_list = []
        obs_rc = [divmod(op, Gw) for op in observed_patch_ids]
        for p in unobserved_patch_ids:
            row, col = divmod(p, Gw)
            dist_sum = 0
            for orow, ocol in obs_rc:
                dist_sum += abs(row - orow) + abs(col - ocol)
            distance_list.append((p, dist_sum))
        distance_list.sort(key=lambda x: x[1])
        unobserved_patch_ids = [p for p, _ in distance_list]

    elif generation_order == "inverse manhattan" and observed_patch_ids:
        inv_list = []
        obs_rc = [divmod(op, Gw) for op in observed_patch_ids]
        for p in unobserved_patch_ids:
            row, col = divmod(p, Gw)
            inv_sum = 0.0
            for orow, ocol in obs_rc:
                dist = abs(row - orow) + abs(col - ocol)
                inv_sum += 1.0 / (dist + 1e-8)
            inv_list.append((p, inv_sum))
        inv_list.sort(key=lambda x: x[1], reverse=True)
        unobserved_patch_ids = [p for p, _ in inv_list]

    elif generation_order == "raster":
        unobserved_patch_ids.sort()

    elif generation_order == "random":
        np.random.shuffle(unobserved_patch_ids)

    else:
        if generation_order not in {"manhattan", "inverse manhattan", "raster", "random"}:
            raise ValueError(f"Unknown generation_order: {generation_order}")

    sampling_order = observed_patch_ids + unobserved_patch_ids

    # log-likelihood prefix sums
    log_likelihoods = np.zeros((batch_size, total_patches))

    counter = 0
    for p_idx in sampling_order:
        with torch.no_grad():
            logits, _ = model(generated_patches, mask_rate=0.0)  # [B, total_patches, V]
        logits_current = logits[:, p_idx, :].clone()

        # Apply conditioning constraints (same for all images)
        if p_idx in conditions_by_patch:
            valid_vocab = torch.ones(model.vocab_size, dtype=torch.bool, device=device)
            for local_idx, val in conditions_by_patch[p_idx]:
                valid_vocab &= (model.vocab[:, local_idx].to(device) == val)
            logits_current[:, ~valid_vocab] = -float('inf')

        # Temperature scaling
        logits_temp = logits_current / float(temperature)
        probs = F.softmax(logits_temp, dim=-1)

        sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]

        # Accumulate log-likelihoods
        for i in range(batch_size):
            lp = math.log(probs[i, sampled_indices[i]].item() + 1e-10)
            log_likelihoods[i, counter] = lp if counter == 0 else (lp + log_likelihoods[i, counter - 1])

        sampled_patches = model.vocab.to(device)[sampled_indices]  # [B, patch_dim]
        generated_patches[:, p_idx, :] = sampled_patches
        counter += 1

    # Reconstruct images
    generated_images = []
    for i in range(batch_size):
        img = patches_to_image(generated_patches[i].detach().cpu(), (H, W), patch_size)
        generated_images.append(img)
    generated_images = torch.stack(generated_images)
    return generated_images, log_likelihoods


def generate_images_batch_method1(
    model,
    patch_size: int,
    image_size,
    batch_size: int,
    device,
    condition_indices=None,
    condition_values=None
):
    """
    Method 1: 'sample-all-then-select' batch generation.

    image_size can be int (square) or (H, W).
    """
    model.eval()
    device = next(model.parameters()).device

    H, W = _parse_image_shape(image_size)
    Gh, Gw, total_patches = _grid_from_shape(H, W, patch_size)
    patch_dim = patch_size * patch_size

    mask_tok = model.mask_token.detach().to(device)
    gen_patches = (
        mask_tok.unsqueeze(0).unsqueeze(0).expand(batch_size, total_patches, patch_dim).clone()
    )

    # build conditioning dict: {patch_idx: [(local_idx, val), ...], ...}
    cond_by_patch = {}
    if condition_indices is not None and condition_values is not None:
        for pix_idx, val in zip(condition_indices, condition_values):
            pix_idx = int(pix_idx)
            row = pix_idx // W
            col = pix_idx % W
            prow, pcol = row // patch_size, col // patch_size
            patch_idx = prow * Gw + pcol
            local_r = row % patch_size
            local_c = col % patch_size
            local_idx = local_r * patch_size + local_c
            cond_by_patch.setdefault(patch_idx, []).append((local_idx, int(val)))
    observed = sorted(cond_by_patch.keys())

    filled = torch.zeros(batch_size, total_patches, dtype=torch.bool, device=device)
    if observed:
        filled[:, observed] = True

    log_liks = [0.0] * batch_size

    # 1) fill observed patches
    if observed:
        with torch.no_grad():
            logits_obs, _ = model(gen_patches, mask_rate=0.0)  # [B, P, V]
        for p in observed:
            lc = logits_obs[:, p, :].clone()
            valid = torch.ones(model.vocab_size, dtype=torch.bool, device=device)
            for (lidx, v) in cond_by_patch[p]:
                valid &= (model.vocab[:, lidx].to(device) == v)
            lc[:, ~valid] = -float('inf')

            probs = F.softmax(lc, dim=-1)
            samp = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]
            for i in range(batch_size):
                log_liks[i] += math.log(probs[i, samp[i]].item() + 1e-10)
            gen_patches[torch.arange(batch_size), p, :] = model.vocab.to(device)[samp]

    # 2) fill remaining patches
    to_fill = total_patches - len(observed)
    for _ in range(to_fill):
        with torch.no_grad():
            logits, _ = model(gen_patches, mask_rate=0.0)  # [B, P, V]

        B, P, V = logits.shape
        flat_logits = logits.view(B * P, V)
        flat_probs = F.softmax(flat_logits, dim=-1)
        flat_samps = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)  # [B*P]
        flat_liks = flat_probs[torch.arange(B * P, device=device), flat_samps]

        samps = flat_samps.view(B, P)
        pliks = flat_liks.view(B, P)

        pliks[filled] = -float('inf')

        choice_patches = torch.argmax(pliks, dim=1)  # [B]
        batch_idx = torch.arange(B, device=device)
        sel_vocab = samps[batch_idx, choice_patches]  # [B]

        for i in range(B):
            log_liks[i] += math.log(pliks[i, choice_patches[i]].item() + 1e-10)

        gen_patches[batch_idx, choice_patches, :] = model.vocab.to(device)[sel_vocab]
        filled[batch_idx, choice_patches] = True

    gen_images = []
    for i in range(batch_size):
        img = patches_to_image(gen_patches[i].detach().cpu(), (H, W), patch_size)
        gen_images.append(img)
    gen_images = torch.stack(gen_images)

    return gen_images, log_liks

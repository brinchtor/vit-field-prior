# -*- coding: utf-8 -*-
"""
Batch sampler that precomputes the forward DP once and reuses it for many draws.
Supports first-order (gamma=0) and second-order (gamma != 0) 2x2 clique models.
"""

import numpy as np

import getQ_2ndorder as getQ


def build_sampler(alf: np.ndarray, bet: float, gamma: float = 0.0):
    """
    Precompute the forward DP once, then return a fast sampler closure.

    Parameters
    ----------
    alf : (n1, n2, d) unary log-potentials.
    bet : scalar axis-aligned coupling.
    gamma : scalar diagonal coupling (0 -> first-order model).

    Returns
    -------
    sample_fn : callable(rng) -> (n1, n2) array of labels.
    logZ : float, exact log partition function.
    """
    ns, z_list, B, logZ = getQ.forward_dp(alf, bet, gamma)

    def sample(rng: np.random.Generator) -> np.ndarray:
        return getQ.backward_sample(alf, bet, gamma, ns, z_list, B, rng)

    return sample, logZ


def generate_samples(
    alf: np.ndarray,
    bet: float,
    N: int,
    gamma: float = 0.0,
    seed: int = 0,
    force_n1_smaller: bool = True,
    return_all: bool = False,
    dtype=np.uint16,
):
    """
    Generate N samples (each n1 x n2).

    If force_n1_smaller and n1 > n2, axes are swapped internally to make the DP
    run on the smaller dimension (which sits in the exponent of the state space).
    Output samples are swapped back to the original orientation.
    """
    alf = np.asarray(alf)
    if alf.ndim != 3:
        raise ValueError("alf must have shape (n1, n2, d).")

    orig_shape = alf.shape
    swapped = False

    if force_n1_smaller and alf.shape[0] > alf.shape[1]:
        alf = np.transpose(alf, (1, 0, 2))
        swapped = True

    n1, n2, d = alf.shape
    sampler, logZ = build_sampler(alf, bet, gamma)

    rng = np.random.default_rng(seed)
    samples = np.empty((N, n1, n2), dtype=dtype)

    for k in range(N):
        q = sampler(rng)
        samples[k] = q.astype(dtype, copy=False)

    if swapped:
        samples = np.transpose(samples, (0, 2, 1))

    if return_all:
        return samples, d, logZ, swapped, orig_shape
    else:
        return samples


def generate_samples_npz(
    alf: np.ndarray,
    bet: float,
    N: int,
    out_path: str,
    gamma: float = 0.0,
    seed: int = 0,
    force_n1_smaller: bool = True,
    dtype=np.uint16,
):
    """
    Generate N samples and save to a compressed npz, including the label-flipped
    copy (assumes binary symmetry; remove the concatenate step if d > 2 or if
    label symmetry doesn't hold for your alf).
    """
    samples, d, logZ, swapped, orig_shape = generate_samples(
        alf, bet, N, gamma=gamma, seed=seed,
        force_n1_smaller=force_n1_smaller, return_all=True, dtype=dtype,
    )

    rng = np.random.default_rng(seed)
    total_samples = np.concatenate([samples, 1 - samples], axis=0)
    shuffled_idx = rng.permutation(len(total_samples))
    total_samples = total_samples[shuffled_idx]

    np.savez_compressed(
        out_path,
        images=total_samples,
        bet=float(bet),
        gamma=float(gamma),
        d=int(d),
        logZ=float(logZ),
        swapped=swapped,
        original_shape=np.array(orig_shape, dtype=int),
        seed=int(seed),
    )


if __name__ == "__main__":
    pass
# -*- coding: utf-8 -*-
"""
Batch sampler that precomputes the forward DP once and reuses it for many draws.
Supports first-order (both gammas = 0) and second-order (any gamma != 0) 2x2
clique models with ANISOTROPIC diagonal couplings.
"""

import numpy as np

import getQ_2ndorder_aniso as getQ


def build_sampler(alf: np.ndarray, bet: float,
                  gamma_ne: float = 0.0, gamma_nw: float = 0.0):
    """
    Precompute the forward DP once, then return a fast sampler closure.

    Parameters
    ----------
    alf : (n1, n2, d) unary log-potentials.
    bet : scalar axis-aligned coupling.
    gamma_ne : scalar coupling on the NE/SW diagonal.
    gamma_nw : scalar coupling on the NW/SE diagonal.

    Returns
    -------
    sample_fn : callable(rng) -> (n1, n2) array of labels.
    logZ : float, exact log partition function.
    """
    ns, z_list, B, logZ = getQ.forward_dp(alf, bet, gamma_ne, gamma_nw)

    def sample(rng: np.random.Generator) -> np.ndarray:
        return getQ.backward_sample(alf, bet, gamma_ne, gamma_nw, ns, z_list, B, rng)

    return sample, logZ


def generate_samples(
    alf: np.ndarray,
    bet: float,
    N: int,
    gamma_ne: float = 0.0,
    gamma_nw: float = 0.0,
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

    NOTE on the axis swap: transposing the lattice (r, c) -> (c, r) maps the
    NE/SW diagonal pair {(r, c), (r-1, c+1)} to {(c, r), (c+1, r-1)}, which in
    the swapped grid has Delta-row = +1 and Delta-col = -1 — i.e. still the
    NE/SW diagonal. Likewise NW/SE stays NW/SE. So the gammas pass through
    unchanged under the axis swap.
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
    sampler, logZ = build_sampler(alf, bet, gamma_ne, gamma_nw)

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
    gamma_ne: float = 0.0,
    gamma_nw: float = 0.0,
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
        alf, bet, N,
        gamma_ne=gamma_ne, gamma_nw=gamma_nw,
        seed=seed,
        force_n1_smaller=force_n1_smaller, return_all=True, dtype=dtype,
    )

    np.savez_compressed(
        out_path,
        images=samples,
        bet=float(bet),
        gamma_ne=float(gamma_ne),
        gamma_nw=float(gamma_nw),
        d=int(d),
        logZ=float(logZ),
        swapped=swapped,
        original_shape=np.array(orig_shape, dtype=int),
        seed=int(seed),
    )


if __name__ == "__main__":
    pass

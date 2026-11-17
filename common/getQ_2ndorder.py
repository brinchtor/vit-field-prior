# -*- coding: utf-8 -*-
"""
Exact sampler for a 2D Markov random field on an (n1 x n2) lattice with
binary or general d-ary labels. Supports both first-order (1x2 and 2x1)
cliques and second-order (2x2) cliques.

Joint:
    P(x) = (1/Z) * prod_i q_i(x_i, neighbors)
with
    log q_i = alpha[i, x_i]
             + bet * 1{x_i = x_{i+1}}        [row < n1]               (down)
             + bet * 1{x_i = x_{i+n1}}       [col < n2]               (right)
             + gamma * 1{x_i = x_{i+n1-1}}   [row > 1 and col < n2]   (NE-diagonal)
             + gamma * 1{x_i = x_{i+n1+1}}   [row < n1 and col < n2]  (SE-diagonal)

Set gamma = 0 to recover the first-order Ising/Potts model.
"""

import numpy as np


# ----------------------------------------------------------------------
# Index <-> vector helpers (unchanged)
# ----------------------------------------------------------------------

def vec2index(vec: np.ndarray, d: int) -> int:
    """Length-nu base-d digits (LSB first) -> 1-based index."""
    vec = np.asarray(vec, dtype=int).ravel()
    idx0 = 0
    p = 1
    for v in vec:
        idx0 += int(v) * p
        p *= d
    return idx0 + 1


def index2vec(index: int, d: int, nu: int) -> np.ndarray:
    """Inverse of vec2index."""
    el = index - 1
    vec = np.zeros(nu, dtype=int)
    for i in range(nu):
        vec[i] = el % d
        el //= d
    return vec


def drawval(pv: np.ndarray, rng: np.random.Generator) -> int:
    """Sample an index from a nonnegative probability vector (will be renormalized)."""
    pv = np.asarray(pv, dtype=float)
    s = pv.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Probability vector has non-positive or invalid sum.")
    pv = pv / s
    cv = np.cumsum(pv)
    u = rng.random()
    return int(np.searchsorted(cv, u, side="right"))


def vec2mat(v: np.ndarray, n1: int) -> np.ndarray:
    """Reshape a length-(n1*n2) column-major vector into (n1, n2)."""
    v = np.asarray(v, dtype=int).ravel()
    n = v.size
    if n % n1 != 0:
        raise ValueError("Length of v must be divisible by n1.")
    n2 = n // n1
    return v.reshape((n1, n2), order="F")


# ----------------------------------------------------------------------
# Local potential q_i
# ----------------------------------------------------------------------

def qeval(l: int, buf: np.ndarray, gi: int, bet: float, gamma: float,
          alf: np.ndarray) -> float:
    """
    Evaluate q_i for label l at site gi, with forward neighbors visible in buf.

    Buffer layout (length n1 + 1 in steady state, shorter near the right edge):
        buf[0]       = site gi+1     (down)
        buf[n1-2]    = site gi+n1-1  (NE-diagonal)
        buf[n1-1]    = site gi+n1    (right)
        buf[n1]      = site gi+n1+1  (SE-diagonal)

    When the buffer is shorter than (n1+1), any neighbor whose buffer index
    is out of range is also out of the grid (handled by the row/col gates).
    """
    n1, n2, d = alf.shape

    col = (gi - 1) // n1 + 1   # 1..n2
    row = (gi - 1) % n1 + 1    # 1..n1

    energy = float(alf[row - 1, col - 1, l])
    B = len(buf)

    # Down neighbor: site gi+1, exists iff row < n1, lives at buf[0]
    if row < n1 and B >= 1:
        if l == buf[0]:
            energy += bet

    # Right neighbor: site gi+n1, exists iff col < n2, lives at buf[n1-1]
    if col < n2 and B >= n1:
        if l == buf[n1 - 1]:
            energy += bet

    # NE-diagonal: site gi+n1-1, exists iff row > 1 and col < n2, lives at buf[n1-2]
    if gamma != 0.0 and row > 1 and col < n2 and B >= n1 - 1:
        if l == buf[n1 - 2]:
            energy += gamma

    # SE-diagonal: site gi+n1+1, exists iff row < n1 and col < n2, lives at buf[n1]
    if gamma != 0.0 and row < n1 and col < n2 and B >= n1 + 1:
        if l == buf[n1]:
            energy += gamma

    return float(np.exp(energy))


# ----------------------------------------------------------------------
# Buffer length B: full-buffer length is n1 for first-order, n1+1 for second
# ----------------------------------------------------------------------

def _full_buffer_length(n1: int, gamma: float) -> int:
    """The DP carries n1 sites of history for first-order, n1+1 for second-order."""
    return n1 + 1 if gamma != 0.0 else n1


# ----------------------------------------------------------------------
# Forward DP (shared by get_q and build_sampler)
# ----------------------------------------------------------------------

def forward_dp(alf: np.ndarray, bet: float, gamma: float = 0.0):
    """
    Run the forward recursion for the model with parameters (alf, bet, gamma).

    Returns
    -------
    ns : np.ndarray of int, shape (n+1,)
        ns[i] is the buffer length at scan position i (1-based).
    z_list : list of np.ndarray
        z_list[i] is the forward normalization vector for scan position i.
    B : int
        Full (steady-state) buffer length.
    logZ : float
        Exact log partition function.
    """
    n1, n2, d = alf.shape
    n = n1 * n2
    B = _full_buffer_length(n1, gamma)
    num0 = d ** B

    ns = np.zeros(n + 1, dtype=int)
    z_list: list[np.ndarray] = [None] * (n + 1)

    for i in range(1, n + 1):
        if i > n - B:
            ns[i] = n - i
            num_i = d ** ns[i]
        else:
            ns[i] = B
            num_i = num0

        if i < n:
            z_i = np.zeros(num_i, dtype=float)
            for k in range(1, num_i + 1):
                xval = index2vec(k, d, ns[i])
                acc = 0.0
                for l in range(d):
                    if i > 1:
                        prev_ns = ns[i - 1]
                        prev_vec = np.concatenate(([l], xval[:prev_ns - 1]))
                        ind = vec2index(prev_vec, d)
                        acc += qeval(l, xval, i, bet, gamma, alf) * z_list[i - 1][ind - 1]
                    else:
                        acc += qeval(l, xval, i, bet, gamma, alf)
                z_i[k - 1] = acc
            z_list[i] = z_i
        else:
            # i == n: collapse — only the unary at site n contributes.
            # z_list[n-1] has length d (one entry per label at site n).
            z_i = np.zeros(1, dtype=float)
            acc = 0.0
            for l in range(d):
                acc += np.exp(alf[n1 - 1, n2 - 1, l]) * z_list[i - 1][l]
            z_i[0] = acc
            z_list[i] = z_i

    Z = z_list[n][0]
    if not np.isfinite(Z) or Z <= 0.0:
        raise ValueError("Partition function is zero/invalid: underflow or impossible evidence.")
    logZ = float(np.log(Z))
    return ns, z_list, B, logZ


# ----------------------------------------------------------------------
# Backward sampling pass (shared by get_q and build_sampler)
# ----------------------------------------------------------------------

def backward_sample(alf: np.ndarray, bet: float, gamma: float,
                    ns: np.ndarray, z_list: list, B: int,
                    rng: np.random.Generator) -> np.ndarray:
    """One backward sweep producing a single (n1, n2) sample."""
    n1, n2, d = alf.shape
    n = n1 * n2

    x = np.zeros(n, dtype=int)
    probvec = np.zeros(d, dtype=float)
    buff = None

    for t in range(1, n + 1):
        j = n - t + 1  # 1-based, going backward

        if j == n:
            denom = z_list[n][0]
            for l in range(d):
                probvec[l] = (np.exp(alf[n1 - 1, n2 - 1, l]) * z_list[j - 1][l]) / denom
            x[j - 1] = drawval(probvec, rng)
        else:
            buffp = buff.copy()
            indbuff = vec2index(buff, d)

            for l in range(d):
                if j > n - B:
                    # Shrinking buffer near the bottom-right corner
                    buffi = np.concatenate(([l], buff[:ns[j]]))
                else:
                    # Full buffer: prepend l, keep B-1 entries from previous buffer
                    buffi = np.concatenate(([l], buff[:ns[j] - 1]))

                indbuffi = vec2index(buffi, d)

                if j == 1:
                    probvec[l] = qeval(l, buff, j, bet, gamma, alf) / z_list[j][indbuff - 1]
                else:
                    probvec[l] = (qeval(l, buff, j, bet, gamma, alf) *
                                  z_list[j - 1][indbuffi - 1] /
                                  z_list[j][indbuff - 1])

            x[j - 1] = drawval(probvec, rng)

        # Update sampled buffer
        if j == n:
            buff = np.array([x[j - 1]], dtype=int)
        elif j > n - B:
            buff = np.concatenate(([x[j - 1]], buffp))
        else:
            buff = np.concatenate(([x[j - 1]], buffp[:B - 1]))

    return vec2mat(x, n1)


# ----------------------------------------------------------------------
# Public entry point: single-shot sampler
# ----------------------------------------------------------------------

def get_q(alf: np.ndarray, bet: float, gamma: float = 0.0,
          rng: np.random.Generator | None = None,
          return_logZ: bool = False):
    """
    Exact sampler for the 2D MRF defined by (alf, bet, gamma).

    Parameters
    ----------
    alf : (n1, n2, d) array of unary log-potentials.
    bet : scalar pairwise (axis-aligned) coupling.
    gamma : scalar diagonal coupling. gamma=0 recovers first-order Ising/Potts.
    rng : optional numpy Generator.
    return_logZ : if True, also return log Z.

    Returns
    -------
    q : (n1, n2) array of labels in {0, ..., d-1}.
    logZ : float (only if return_logZ).
    """
    if rng is None:
        rng = np.random.default_rng()

    ns, z_list, B, logZ = forward_dp(alf, bet, gamma)
    q = backward_sample(alf, bet, gamma, ns, z_list, B, rng)

    if return_logZ:
        return q, logZ
    return q
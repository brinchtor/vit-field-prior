# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 12:51:02 2026

@author: brinc
"""

import numpy as np


def vec2index(vec: np.ndarray, d: int) -> int:
    """
    Map a vector of base-d digits (least-significant digit first) to a 1-based index.
    vec entries must be in {0, ..., d-1}.
    """
    vec = np.asarray(vec, dtype=int).ravel()
    idx0 = 0
    p = 1
    for v in vec:
        idx0 += int(v) * p
        p *= d
    return idx0 + 1  # 1-based like MATLAB


def index2vec(index: int, d: int, nu: int) -> np.ndarray:
    """
    Inverse of vec2index: 1-based index -> length-nu vector of digits base d,
    least-significant digit first.
    """
    el = index - 1
    vec = np.zeros(nu, dtype=int)
    for i in range(nu):
        vec[i] = el % d
        el //= d
    return vec


def qeval(l: int, buf: np.ndarray, gi: int, bet: float, alf: np.ndarray) -> float:
    n1, n2, d = alf.shape

    col = (gi - 1) // n1 + 1   # 1..n2
    row = (gi - 1) % n1 + 1    # 1..n1

    eq = 0

    # down neighbor exists if not bottom row
    if row < n1:
        # down neighbor is next in scan -> buf[0]
        eq += (l == buf[0])

    # right neighbor exists if not last column
    if col < n2:
        # right neighbor is exactly n1 steps ahead in scan -> buf[n1-1]
        eq += (l == buf[n1 - 1])

    return float(np.exp(bet * eq + alf[row - 1, col - 1, l]))


def drawval(pv: np.ndarray, rng: np.random.Generator) -> int:
    """
    Sample an index from a discrete distribution pv (assumed nonnegative and sum>0).
    Returns an integer in {0..len(pv)-1}.
    """
    pv = np.asarray(pv, dtype=float)
    s = pv.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Probability vector has non-positive or invalid sum.")
    pv = pv / s
    cv = np.cumsum(pv)
    u = rng.random()
    return int(np.searchsorted(cv, u, side="right"))


def vec2mat(v: np.ndarray, n1: int) -> np.ndarray:
    """
    Reshape a length-(n1*n2) vector laid out in column-major order into (n1, n2).
    """
    v = np.asarray(v, dtype=int).ravel()
    n = v.size
    if n % n1 != 0:
        raise ValueError("Length of v must be divisible by n1.")
    n2 = n // n1
    return v.reshape((n1, n2), order="F")


def get_q(alf: np.ndarray, bet: float, rng: np.random.Generator | None = None,
          return_logZ: bool = False):
    """
    Exact sampler for the grid Potts/Markov random field defined by unary log-potentials alf and coupling bet.

    alf: shape (n1, n2, d), unary log-potentials
    bet: scalar interaction parameter
    rng: numpy random generator (optional)
    return_logZ: if True, also returns log partition function estimate from DP (exact for this model)

    Returns
    -------
    q: array shape (n1, n2) with entries in {0..d-1}
    (optional) logZ: float
    """
    if rng is None:
        rng = np.random.default_rng()

    n1, n2, d = alf.shape
    n = n1 * n2
    num0 = d ** n1  # maximum buffer configurations

    # Store buffer size ns[i] and forward constants z_list[i]
    # Using 1-based indexing for i to match the original recursion
    ns = np.zeros(n + 1, dtype=int)
    num = np.zeros(n + 1, dtype=int)
    z_list: list[np.ndarray] = [None] * (n + 1)  # z_list[i] is array of length num[i]

    # ---------- Forward recursion ----------
    for i in range(1, n + 1):
        if i > n - n1:
            ns[i] = n - i
            num[i] = d ** ns[i]
        else:
            ns[i] = n1
            num[i] = num0

        if i < n:
            z_i = np.zeros(num[i], dtype=float)
            for k in range(1, num[i] + 1):
                xval = index2vec(k, d, ns[i])  # 0..d-1
                acc = 0.0
                for l in range(d):
                    if i > 1:
                        prev_ns = ns[i - 1]
                        prev_vec = np.concatenate(([l], xval[:prev_ns - 1]))
                        ind = vec2index(prev_vec, d)
                        acc += qeval(l, xval, i, bet, alf) * z_list[i - 1][ind - 1]
                    else:
                        acc += qeval(l, xval, i, bet, alf)
                z_i[k - 1] = acc
            z_list[i] = z_i
        else:
            # i == n: final normalization collapse
            z_i = np.zeros(1, dtype=float)
            acc = 0.0
            # z_list[n-1] should have length d at this point
            for l in range(d):
                acc += np.exp(alf[n1 - 1, n2 - 1, l]) * z_list[i - 1][l]
            z_i[0] = acc
            z_list[i] = z_i

    logZ = float(np.log(z_list[n][0]))

    # ---------- Backward sampling ----------
    x = np.zeros(n, dtype=int)
    probvec = np.zeros(d, dtype=float)
    buff = None  # will be a 1D array of labels (0..d-1)

    for t in range(1, n + 1):
        j = n - t + 1  # 1-based node index going backward

        if j == n:
            denom = z_list[n][0]
            for l in range(d):
                probvec[l] = (np.exp(alf[n1 - 1, n2 - 1, l]) * z_list[j - 1][l]) / denom
            x[j - 1] = drawval(probvec, rng)
        else:
            buffp = buff.copy()
            indbuff = vec2index(buff, d)

            for l in range(d):
                if j > n - n1:
                    # shrinking buffer in last column
                    buffi = np.concatenate(([l], buff[:ns[j]]))
                else:
                    # full buffer: keep n1-1 entries from previous buffer
                    buffi = np.concatenate(([l], buff[:ns[j] - 1]))

                indbuffi = vec2index(buffi, d)

                if j == 1:
                    probvec[l] = qeval(l, buff, j, bet, alf) / z_list[j][indbuff - 1]
                else:
                    probvec[l] = (qeval(l, buff, j, bet, alf) *
                                  z_list[j - 1][indbuffi - 1] /
                                  z_list[j][indbuff - 1])

            x[j - 1] = drawval(probvec, rng)

        # Update sampled buffer
        if j == n:
            buff = np.array([x[j - 1]], dtype=int)
        elif j > n - n1:
            buff = np.concatenate(([x[j - 1]], buffp))
        else:
            buff = np.concatenate(([x[j - 1]], buffp[:n1 - 1]))

    q = vec2mat(x, n1)

    if return_logZ:
        return q, logZ
    return q

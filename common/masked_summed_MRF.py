# -*- coding: utf-8 -*-
"""
Exact conditioning on a SINGLE summed observation over one full column ("well").

Observation model: we observe d = sum_r x_{r, c*} for one column c*, with x in {0,1}.
Because the observed set is exactly one buffer-width column, the column sum is a
deterministic function of the forward buffer state at the single scan position
where the frontier buffer coincides with column c*. Conditioning is therefore
ONE indicator mask applied to the forward message at that step -- no count
augmentation, no enumeration of the C(n1, d) matching configurations. Cost is the
same order as an ordinary exact run.

Four public functions:
  generate_MRFs_masked                     -- exact conditional samples, 1st order
  generate_MRFs_masked_2ndorder_aniso      -- exact conditional samples, 2nd order
  exact_marginals_masked                   -- exact posterior marginals, 1st order
  exact_marginals_masked_2ndorder_aniso    -- exact posterior marginals, 2nd order

Conventions (inherited from getQ*.py / exact_marginals*.py):
  * column-major scan; alf has shape (n1, n2, 2); labels in {0, 1}.
  * The well is a column along the n1 (buffer) axis. NO axis swap is applied here,
    so pass alf with n1 <= n2 (for a 12x24 grid this already holds). obs_col is the
    0-based column index of the well and must satisfy 1 <= obs_col <= n2-1 (column 0
    is unsupported because its frontier-buffer alignment falls at scan index 0).
  * obs_sum is the observed number of 1s in the well, in [0, n1].

The samplers reuse getQ_2ndorder_aniso.backward_sample (which reduces exactly to the
first-order backward pass at gamma_ne = gamma_nw = 0); the marginals reuse the
state-map / local-factor helpers from exact_marginals*.py. Only the forward pass is
re-implemented, so the column-sum mask can be injected mid-recursion.
"""

from __future__ import annotations

import numpy as np

import getQ_2ndorder_aniso as getQ2
import exact_marginals as em1
import exact_marginals_2ndorder_aniso as em2


# ======================================================================
# Shared helpers
# ======================================================================
def _popcount(arr: np.ndarray) -> np.ndarray:
    """Vectorised bit population count for nonnegative integers."""
    a = arr.astype(np.int64).copy()
    c = np.zeros_like(a)
    while a.any():
        c += a & 1
        a >>= 1
    return c


def _check_well(obs_col: int, obs_sum: int, n1: int, n2: int) -> None:
    if obs_col < 1 or obs_col > n2 - 1:
        raise ValueError(
            f"obs_col (0-based) must be in [1, {n2 - 1}]; column 0 is unsupported "
            "-- use an interior or last column for the well."
        )
    if obs_sum < 0 or obs_sum > n1:
        raise ValueError(f"obs_sum must be in [0, {n1}] (number of 1s in the column).")


def _star_index(obs_col: int, n1: int) -> int:
    """Scan index i* at which the forward frontier buffer equals column obs_col."""
    return obs_col * n1


# ======================================================================
# Masked forward DP (shared by both samplers; gammas=0 -> first order)
# ======================================================================
def _masked_forward(alf, bet, gamma_ne, gamma_nw, i_star, obs_sum):
    """
    Re-implementation of getQ_2ndorder_aniso.forward_dp with a single-column sum
    mask applied to the forward message at scan step i_star. Returns (ns, z_list,
    B, logZ_cond), where z_list is the masked (unnormalised) forward table and
    logZ_cond = log of the conditioned partition mass (= logZ + log P(sum = obs_sum)).
    """
    n1, n2, d = alf.shape
    n = n1 * n2
    B = getQ2._full_buffer_length(n1, gamma_ne, gamma_nw)
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
                xval = getQ2.index2vec(k, d, ns[i])
                acc = 0.0
                for l in range(d):
                    fac = getQ2.qeval(l, xval, i, bet, gamma_ne, gamma_nw, alf)
                    if i > 1:
                        prev = np.concatenate(([l], xval[: ns[i - 1] - 1]))
                        acc += fac * z_list[i - 1][getQ2.vec2index(prev, d) - 1]
                    else:
                        acc += fac
                z_i[k - 1] = acc

            # --- single-column sum mask: keep only buffer configs whose column
            #     (first n1 buffer slots) has the observed number of 1s ---
            if i == i_star:
                for k in range(1, num_i + 1):
                    vec = getQ2.index2vec(k, d, ns[i])
                    if int(np.sum(vec[:n1])) != obs_sum:
                        z_i[k - 1] = 0.0

            z_list[i] = z_i
        else:
            z_i = np.zeros(1, dtype=float)
            acc = 0.0
            for l in range(d):
                acc += np.exp(alf[n1 - 1, n2 - 1, l]) * z_list[i - 1][l]
            z_i[0] = acc
            z_list[i] = z_i

    Z = z_list[n][0]
    if not np.isfinite(Z) or Z <= 0.0:
        raise ValueError(
            "Masked partition function is zero/invalid: obs_sum is infeasible "
            "for this column, or numerical underflow."
        )
    return ns, z_list, B, float(np.log(Z))


# ======================================================================
# Samplers
# ======================================================================
def generate_MRFs_masked(alf, bet, N, obs_col, obs_sum, seed=0, dtype=np.uint8):
    """
    Exact conditional samples from the first-order MRF given sum(column obs_col) = obs_sum.
    Returns samples of shape (N, n1, n2).
    """
    alf = np.asarray(alf, dtype=float)
    if alf.ndim != 3 or alf.shape[2] != 2:
        raise ValueError("alf must have shape (n1, n2, 2).")
    n1, n2, _ = alf.shape
    if n1 > n2:
        raise ValueError("Pass alf with n1 <= n2 (the well is a column along n1; no swap is applied).")
    _check_well(obs_col, obs_sum, n1, n2)

    i_star = _star_index(obs_col, n1)
    ns, z_list, B, _ = _masked_forward(alf, bet, 0.0, 0.0, i_star, obs_sum)

    rng = np.random.default_rng(seed)
    out = np.empty((N, n1, n2), dtype=dtype)
    for k in range(N):
        q = getQ2.backward_sample(alf, bet, 0.0, 0.0, ns, z_list, B, rng)
        out[k] = q.astype(dtype, copy=False)
    return out


def generate_MRFs_masked_2ndorder_aniso(alf, bet, N, obs_col, obs_sum,
                                        gamma_ne=0.0, gamma_nw=0.0,
                                        seed=0, dtype=np.uint8):
    """
    Exact conditional samples from the second-order anisotropic MRF given
    sum(column obs_col) = obs_sum. Returns samples of shape (N, n1, n2).
    """
    alf = np.asarray(alf, dtype=float)
    if alf.ndim != 3 or alf.shape[2] != 2:
        raise ValueError("alf must have shape (n1, n2, 2).")
    n1, n2, _ = alf.shape
    if n1 > n2:
        raise ValueError("Pass alf with n1 <= n2 (the well is a column along n1; no swap is applied).")
    _check_well(obs_col, obs_sum, n1, n2)
    gamma_ne = float(gamma_ne); gamma_nw = float(gamma_nw)

    i_star = _star_index(obs_col, n1)
    ns, z_list, B, _ = _masked_forward(alf, bet, gamma_ne, gamma_nw, i_star, obs_sum)

    rng = np.random.default_rng(seed)
    out = np.empty((N, n1, n2), dtype=dtype)
    for k in range(N):
        q = getQ2.backward_sample(alf, bet, gamma_ne, gamma_nw, ns, z_list, B, rng)
        out[k] = q.astype(dtype, copy=False)
    return out


# ======================================================================
# Exact marginals (fast path, with the column-sum mask)
# ======================================================================
def _marginals_fast_masked_1st(alf, bet, obs_scan, i_star, obs_sum):
    n1, n2, d = alf.shape
    assert d == 2
    n = n1 * n2
    ns = em1._buffer_schedule(n, n1)
    S = 1 << n1
    slot0, slotR, parent = em1._state_maps(n1)
    colcount = _popcount(np.arange(S, dtype=np.int64) & ((1 << n1) - 1))

    # ---- forward ----
    F = [None] * (n + 1)
    logscale = np.zeros(n + 1)
    for i in range(1, n + 1):
        col = (i - 1) // n1 + 1
        row = (i - 1) % n1 + 1
        q = np.exp(em1._logq_arr(alf, bet, row, col, n1, n2, slot0, slotR))
        want = obs_scan.get(i)
        if want is not None:
            q[:, 1 - want] = 0.0
        if i == 1:
            Fi = q[:, 0] + q[:, 1]
        else:
            Fp = F[i - 1]
            Fi = q[:, 0] * Fp[parent[:, 0]] + q[:, 1] * Fp[parent[:, 1]]
        if i == i_star:
            Fi = Fi * (colcount == obs_sum)
        sc = Fi.max()
        if not np.isfinite(sc) or sc <= 0:
            raise ValueError(f"forward mass invalid at step {i} (obs_sum infeasible?).")
        F[i] = Fi / sc
        logscale[i] = logscale[i - 1] + np.log(sc)

    logZ = float(logscale[n] + np.log(F[n].sum()) - n1 * np.log(2.0))

    # ---- backward ----
    marg = np.zeros((n1, n2, d))
    D = np.zeros(S); D[0] = 1.0
    for t in range(1, n + 1):
        j = n - t + 1
        col = (j - 1) // n1 + 1
        row = (j - 1) % n1 + 1
        q = np.exp(em1._logq_arr(alf, bet, row, col, n1, n2, slot0, slotR))
        want = obs_scan.get(j)
        if want is not None:
            q[:, 1 - want] = 0.0
        if j == 1:
            num0 = q[:, 0]; num1 = q[:, 1]
        else:
            Fp = F[j - 1]
            num0 = q[:, 0] * Fp[parent[:, 0]]
            num1 = q[:, 1] * Fp[parent[:, 1]]
        denom = num0 + num1
        safe = denom > 0
        c0 = np.zeros(S); c1 = np.zeros(S)
        c0[safe] = num0[safe] / denom[safe]
        c1[safe] = num1[safe] / denom[safe]
        m0 = float((D * c0).sum()); m1 = float((D * c1).sum())
        tot = m0 + m1
        marg[row - 1, col - 1, 0] = m0 / tot
        marg[row - 1, col - 1, 1] = m1 / tot
        Dn = np.zeros(S)
        np.add.at(Dn, parent[:, 0], D * c0)
        np.add.at(Dn, parent[:, 1], D * c1)
        sD = Dn.sum()
        if sD <= 0 or not np.isfinite(sD):
            raise ValueError(f"distribution mass invalid at site {j}.")
        D = Dn / sD
    return marg, logZ


def _marginals_fast_masked_2nd(alf, bet, gamma_ne, gamma_nw, obs_scan, i_star, obs_sum):
    n1, n2, d = alf.shape
    assert d == 2
    n = n1 * n2
    B = em2._full_buffer_length(n1, gamma_ne, gamma_nw)
    ns = em2._buffer_schedule(n, B)
    S = 1 << B
    slot0, slotR, slotNE, slotNW, parent = em2._state_maps(n1, B)
    colcount = _popcount(np.arange(S, dtype=np.int64) & ((1 << n1) - 1))

    def qarr(row, col):
        return np.exp(em2._logq_arr(alf, bet, gamma_ne, gamma_nw, row, col,
                                    n1, n2, slot0, slotR, slotNE, slotNW))

    # ---- forward ----
    F = [None] * (n + 1)
    logscale = np.zeros(n + 1)
    for i in range(1, n + 1):
        col = (i - 1) // n1 + 1
        row = (i - 1) % n1 + 1
        q = qarr(row, col)
        want = obs_scan.get(i)
        if want is not None:
            q[:, 1 - want] = 0.0
        if i == 1:
            Fi = q[:, 0] + q[:, 1]
        else:
            Fp = F[i - 1]
            Fi = q[:, 0] * Fp[parent[:, 0]] + q[:, 1] * Fp[parent[:, 1]]
        if i == i_star:
            Fi = Fi * (colcount == obs_sum)
        sc = Fi.max()
        if not np.isfinite(sc) or sc <= 0:
            raise ValueError(f"forward mass invalid at step {i} (obs_sum infeasible?).")
        F[i] = Fi / sc
        logscale[i] = logscale[i - 1] + np.log(sc)

    logZ = float(logscale[n] + np.log(F[n].sum()) - B * np.log(2.0))

    # ---- backward ----
    marg = np.zeros((n1, n2, d))
    D = np.zeros(S); D[0] = 1.0
    for t in range(1, n + 1):
        j = n - t + 1
        col = (j - 1) // n1 + 1
        row = (j - 1) % n1 + 1
        q = qarr(row, col)
        want = obs_scan.get(j)
        if want is not None:
            q[:, 1 - want] = 0.0
        if j == 1:
            num0 = q[:, 0]; num1 = q[:, 1]
        else:
            Fp = F[j - 1]
            num0 = q[:, 0] * Fp[parent[:, 0]]
            num1 = q[:, 1] * Fp[parent[:, 1]]
        denom = num0 + num1
        safe = denom > 0
        c0 = np.zeros(S); c1 = np.zeros(S)
        c0[safe] = num0[safe] / denom[safe]
        c1[safe] = num1[safe] / denom[safe]
        m0 = float((D * c0).sum()); m1 = float((D * c1).sum())
        tot = m0 + m1
        marg[row - 1, col - 1, 0] = m0 / tot
        marg[row - 1, col - 1, 1] = m1 / tot
        Dn = np.zeros(S)
        np.add.at(Dn, parent[:, 0], D * c0)
        np.add.at(Dn, parent[:, 1], D * c1)
        sD = Dn.sum()
        if sD <= 0 or not np.isfinite(sD):
            raise ValueError(f"distribution mass invalid at site {j}.")
        D = Dn / sD
    return marg, logZ


def exact_marginals_masked(alf, bet, obs_col, obs_sum,
                           observations=None, return_logZ=False):
    """
    Exact per-site posterior marginals given sum(column obs_col) = obs_sum
    (first-order MRF). observations is an optional dict {(r,c): label} of extra
    hard per-pixel constraints (0-based, original orientation). No axis swap.
    """
    alf = np.asarray(alf, dtype=float)
    if alf.ndim != 3 or alf.shape[2] != 2:
        raise ValueError("alf must have shape (n1, n2, 2).")
    n1, n2, _ = alf.shape
    if n1 > n2:
        raise ValueError("Pass alf with n1 <= n2 (the well is a column along n1; no swap is applied).")
    _check_well(obs_col, obs_sum, n1, n2)

    obs = dict(observations) if observations else {}
    obs_scan = em1._obs_by_scan(obs, n1, n2)
    i_star = _star_index(obs_col, n1)
    marg, logZ = _marginals_fast_masked_1st(alf, bet, obs_scan, i_star, obs_sum)
    return (marg, logZ) if return_logZ else marg


def exact_marginals_masked_2ndorder_aniso(alf, bet, obs_col, obs_sum,
                                          gamma_ne=0.0, gamma_nw=0.0,
                                          observations=None, return_logZ=False):
    """
    Exact per-site posterior marginals given sum(column obs_col) = obs_sum
    (second-order anisotropic MRF). No axis swap.
    """
    alf = np.asarray(alf, dtype=float)
    if alf.ndim != 3 or alf.shape[2] != 2:
        raise ValueError("alf must have shape (n1, n2, 2).")
    n1, n2, _ = alf.shape
    if n1 > n2:
        raise ValueError("Pass alf with n1 <= n2 (the well is a column along n1; no swap is applied).")
    _check_well(obs_col, obs_sum, n1, n2)
    gamma_ne = float(gamma_ne); gamma_nw = float(gamma_nw)

    obs = dict(observations) if observations else {}
    obs_scan = em2._obs_by_scan(obs, n1, n2)
    i_star = _star_index(obs_col, n1)
    marg, logZ = _marginals_fast_masked_2nd(alf, bet, gamma_ne, gamma_nw,
                                            obs_scan, i_star, obs_sum)
    return (marg, logZ) if return_logZ else marg


# ======================================================================
# Sum-filtered brute force + self-test
# ======================================================================
def _bruteforce_colsum(alf, bet, obs_col, obs_sum, gamma_ne=0.0, gamma_nw=0.0):
    n1, n2, d = alf.shape
    n = n1 * n2
    marg = np.zeros((n1, n2, d)); Z = 0.0
    for code in range(d ** n):
        x = np.zeros((n1, n2), dtype=int); c = code
        for gi in range(1, n + 1):
            col = (gi - 1) // n1; row = (gi - 1) % n1
            x[row, col] = c % d; c //= d
        if int(x[:, obs_col].sum()) != obs_sum:
            continue
        w = 0.0
        for col in range(n2):
            for row in range(n1):
                l = x[row, col]; w += alf[row, col, l]
                if row < n1 - 1 and x[row + 1, col] == l: w += bet
                if col < n2 - 1 and x[row, col + 1] == l: w += bet
                if row > 0 and col < n2 - 1 and x[row - 1, col + 1] == l: w += gamma_ne
                if row < n1 - 1 and col < n2 - 1 and x[row + 1, col + 1] == l: w += gamma_nw
        w = np.exp(w); Z += w
        for col in range(n2):
            for row in range(n1):
                marg[row, col, x[row, col]] += w
    return marg / Z, float(np.log(Z))


def _selftest():
    rng = np.random.default_rng(0)
    print("=" * 70)
    print("Masked single-column-sum: fast marginals vs sum-filtered brute force")
    cases = [
        # (n1, n2, bet, gne, gnw, obs_col, obs_sum)
        (3, 4, 0.7, 0.0, 0.0, 1, 2),
        (3, 4, 0.7, 0.0, 0.0, 3, 1),   # last column
        (4, 4, 0.5, 0.0, 0.0, 2, 3),
        (3, 4, 0.5, 0.3, 0.1, 1, 2),   # second order
        (3, 4, 0.5, 0.4, -0.2, 3, 0),  # second order, last col, sum 0
        (3, 5, 0.4, 0.2, 0.0, 3, 2),   # ne only
    ]
    for (n1, n2, bet, gne, gnw, oc, osum) in cases:
        a = rng.normal(size=(n1, n2, 2)) * 0.3
        if gne == 0.0 and gnw == 0.0:
            mf, lzf = exact_marginals_masked(a, bet, oc, osum, return_logZ=True)
        else:
            mf, lzf = exact_marginals_masked_2ndorder_aniso(
                a, bet, oc, osum, gamma_ne=gne, gamma_nw=gnw, return_logZ=True)
        mb, lzb = _bruteforce_colsum(a, bet, oc, osum, gamma_ne=gne, gamma_nw=gnw)
        # column marginals must respect the sum exactly
        colmean = mf[:, oc, 1].sum()
        print(f"  n1={n1} n2={n2} g=({gne:+.1f},{gnw:+.1f}) col={oc} sum={osum}: "
              f"marg-bf={np.abs(mf - mb).max():.2e}  logZ-bf={abs(lzf - lzb):.2e}  "
              f"E[colsum]={colmean:.4f} (target {osum})")

    print("=" * 70)
    print("Masked samplers: MC sample-mean vs exact masked marginals")
    for (n1, n2, bet, gne, gnw, oc, osum, S) in [
        (3, 4, 0.7, 0.0, 0.0, 1, 2, 60000),
        (3, 4, 0.5, 0.3, 0.1, 2, 1, 60000),
    ]:
        a = rng.normal(size=(n1, n2, 2)) * 0.3
        if gne == 0.0 and gnw == 0.0:
            me = exact_marginals_masked(a, bet, oc, osum)[..., 1]
            smp = generate_MRFs_masked(a, bet, S, oc, osum, seed=1)
        else:
            me = exact_marginals_masked_2ndorder_aniso(
                a, bet, oc, osum, gamma_ne=gne, gamma_nw=gnw)[..., 1]
            smp = generate_MRFs_masked_2ndorder_aniso(
                a, bet, S, oc, osum, gamma_ne=gne, gamma_nw=gnw, seed=1)
        # every sample must hit the column sum exactly
        colsum_ok = np.all(smp[:, :, oc].sum(axis=1) == osum)
        mc = smp.mean(axis=0)
        print(f"  n1={n1} n2={n2} g=({gne:+.1f},{gnw:+.1f}) col={oc} sum={osum}: "
              f"max|exact-MC({S})|={np.abs(me - mc).max():.2e} (~{1/np.sqrt(S):.1e})  "
              f"all-samples-hit-sum={colsum_ok}")


if __name__ == "__main__":
    _selftest()
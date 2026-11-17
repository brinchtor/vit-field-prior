# -*- coding: utf-8 -*-
"""
Exact per-site posterior marginals (and the posterior expected-value field) for
the second-order ANISOTROPIC binary grid MRF defined in getQ_2ndorder_aniso.py.

This is the second-order counterpart of exact_marginals.py: where that file
mirrors getQ.py (first-order Ising/Potts, axis-aligned couplings only), this one
mirrors getQ_2ndorder_aniso.py, which adds two diagonal couplings,

    gamma_ne : NE/SW diagonal,  site i <-> site i+n1-1   (row>1 and col<n2)
    gamma_nw : NW/SE diagonal,  site i <-> site i+n1+1   (row<n1 and col<n2)

on top of the axis-aligned bet (down + right). Setting gamma_ne = gamma_nw = 0
recovers the first-order model exactly, and both code paths below collapse to the
first-order recursion in that case.

Two code paths, both exact:
  * method="reference"  : dict-based, calls getQ_2ndorder_aniso.qeval /
                          .vec2index / .index2vec directly. Slow but a
                          transparent oracle. Validated against brute-force
                          enumeration on small grids.
  * method="fast" (default for d=2): a faithful VECTORISATION of the reference
                          recursion. Precomputes the buffer-index bookkeeping once
                          with numpy so the per-site work is array arithmetic and
                          fancy-index gathers.

The fast path is checked against the reference path inside the self-tests, so the
two agree to machine precision before either is trusted at thesis scale.

Conventions are inherited verbatim from getQ_2ndorder_aniso.py (column-major scan;
each site pays its DOWN edge = buffer slot 0, RIGHT edge = buffer slot n1-1,
NE/SW edge = buffer slot n1-2, NW/SE edge = buffer slot n1, plus its unary). The
forward "history" buffer therefore has full width

    B = n1 + 1   if (gamma_ne != 0 or gamma_nw != 0)   [second order]
    B = n1       otherwise                              [first order]

exactly as getQ_2ndorder_aniso._full_buffer_length. Observations are folded in as
hard per-site label constraints, giving exact posterior marginals P(x_i | y);
observed sites read 0/1.
"""

from __future__ import annotations

import numpy as np
import getQ_2ndorder_aniso as getQ


# ======================================================================
# Shared helpers
# ======================================================================
def _full_buffer_length(n1: int, gamma_ne: float, gamma_nw: float) -> int:
    """Same rule as getQ_2ndorder_aniso: n1+1 if any diagonal is active, else n1."""
    return n1 + 1 if (gamma_ne != 0.0 or gamma_nw != 0.0) else n1


def _buffer_schedule(n: int, B: int) -> np.ndarray:
    ns = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        ns[i] = (n - i) if i > n - B else B
    return ns


def _obs_by_scan(obs, n1, n2):
    d = {}
    for (r0, c0), v in obs.items():
        if not (0 <= r0 < n1 and 0 <= c0 < n2):
            raise ValueError(f"obs coord {(r0, c0)} out of range for {(n1, n2)}.")
        d[c0 * n1 + r0 + 1] = int(v)  # 1-based scan index
    return d


# ======================================================================
# REFERENCE path (dict-based oracle, validated against brute force)
# ======================================================================
def _absorbed_site(i, n):
    return i  # qeval(l, xval, i, ...) applies the unary at site i; l is site i's value


def _forward_table_ref(alf, bet, gamma_ne, gamma_nw, ns, mask):
    n1, n2, d = alf.shape
    n = n1 * n2
    F = [None] * (n + 1)
    logscale = np.zeros(n + 1)
    for i in range(1, n + 1):
        num_i = d ** ns[i]
        if i < n:
            Fi = np.zeros(num_i)
            for k in range(1, num_i + 1):
                xval = getQ.index2vec(k, d, ns[i])
                acc = 0.0
                for l in range(d):
                    if mask is not None and not mask(_absorbed_site(i, n), l):
                        continue
                    fac = getQ.qeval(l, xval, i, bet, gamma_ne, gamma_nw, alf)
                    if i > 1:
                        prev = np.concatenate(([l], xval[: ns[i - 1] - 1]))
                        acc += fac * F[i - 1][getQ.vec2index(prev, d) - 1]
                    else:
                        acc += fac
                Fi[k - 1] = acc
            F[i] = Fi
        else:
            acc = 0.0
            for l in range(d):
                if mask is not None and not mask(n, l):
                    continue
                acc += np.exp(alf[n1 - 1, n2 - 1, l]) * F[i - 1][l]
            F[i] = np.array([acc])
        s = F[i].sum()
        if not np.isfinite(s) or s <= 0:
            raise ValueError(f"forward mass invalid at step {i}.")
        F[i] = F[i] / s
        logscale[i] = logscale[i - 1] + np.log(s)
    return F, logscale


def _marginals_ref(alf, bet, gamma_ne, gamma_nw, obs_by_scan):
    n1, n2, d = alf.shape
    n = n1 * n2
    B = _full_buffer_length(n1, gamma_ne, gamma_nw)
    ns = _buffer_schedule(n, B)
    mask = (lambda gi, l: obs_by_scan.get(gi, l) == l) if obs_by_scan else None
    F, logscale = _forward_table_ref(alf, bet, gamma_ne, gamma_nw, ns, mask)
    logZ = float(logscale[n])

    marg = np.zeros((n1, n2, d))
    D = {(): 1.0}
    for t in range(1, n + 1):
        j = n - t + 1
        col = (j - 1) // n1 + 1
        row = (j - 1) % n1 + 1
        newD = {}
        site_mass = np.zeros(d)
        for buff_tuple, pw in D.items():
            buff = np.array(buff_tuple, dtype=int)
            pv = np.zeros(d)
            if j == n:
                for l in range(d):
                    if mask and not mask(n, l):
                        continue
                    pv[l] = np.exp(alf[n1 - 1, n2 - 1, l]) * F[j - 1][l]
            else:
                for l in range(d):
                    if mask and not mask(j, l):
                        continue
                    if j > n - B:
                        buffi = np.concatenate(([l], buff[: ns[j]]))
                    else:
                        buffi = np.concatenate(([l], buff[: ns[j] - 1]))
                    fac = getQ.qeval(l, buff, j, bet, gamma_ne, gamma_nw, alf)
                    if j == 1:
                        pv[l] = fac
                    else:
                        pv[l] = fac * F[j - 1][getQ.vec2index(buffi, d) - 1]
            ssum = pv.sum()
            pv = pv / ssum
            for l in range(d):
                if pv[l] == 0:
                    continue
                contrib = pw * pv[l]
                site_mass[l] += contrib
                if j == n:
                    nb = (l,)
                elif j > n - B:
                    nb = (l,) + buff_tuple
                else:
                    nb = (l,) + buff_tuple[: B - 1]
                newD[nb] = newD.get(nb, 0.0) + contrib
        marg[row - 1, col - 1, :] = site_mass
        D = newD
    marg = marg / marg.sum(axis=2, keepdims=True)
    return marg, logZ


# ======================================================================
# FAST path: vectorised faithful copy of the reference recursion (d=2)
# ======================================================================
def _state_maps(n1, B):
    """
    Precompute, for all full-width buffer states s in [0, 2^B):
      slot0[s]   : bit 0      (down  neighbour, site i+1)
      slotR[s]   : bit n1-1   (right neighbour, site i+n1)
      slotNE[s]  : bit n1-2   (NE/SW neighbour, site i+n1-1)   [needs n1 >= 2]
      slotNW[s]  : bit n1     (NW/SE neighbour, site i+n1+1)   [needs B >= n1+1]
      parent[s,l]: index of [l, s_slots[:B-1]]  (prepend l, drop slot B-1),
                   i.e. getQ's prev/buffi map in the bulk.

    slotNE/slotNW are returned as None when the corresponding slot is outside the
    buffer; the caller only touches them when the matching gamma is nonzero, which
    is exactly when the slot is in range.
    """
    S = 1 << B
    s = np.arange(S, dtype=np.int64)
    slot0 = (s & 1).astype(np.int64)
    slotR = ((s >> (n1 - 1)) & 1).astype(np.int64)
    slotNE = ((s >> (n1 - 2)) & 1).astype(np.int64) if n1 >= 2 else None
    slotNW = ((s >> n1) & 1).astype(np.int64) if B >= n1 + 1 else None
    low = s & ((1 << (B - 1)) - 1)               # slots 0..B-2
    parent = np.empty((S, 2), dtype=np.int64)
    parent[:, 0] = (low << 1) | 0
    parent[:, 1] = (low << 1) | 1
    return slot0, slotR, slotNE, slotNW, parent


def _logq_arr(alf, bet, gamma_ne, gamma_nw, row, col, n1, n2,
              slot0, slotR, slotNE, slotNW):
    """log local factor[s,l] for site (row,col); shape (S,2).

    Off-lattice neighbours are dropped purely via the row/col gates: in the
    column-major scan, whenever a neighbour is on the grid the buffer is already
    long enough to hold its slot, so no explicit buffer-length test is needed
    (the same reason the first-order fast path only checks has_down/has_right).
    """
    has_down = row < n1
    has_right = col < n2
    has_ne = (gamma_ne != 0.0) and (row > 1) and (col < n2)
    has_nw = (gamma_nw != 0.0) and (row < n1) and (col < n2)
    out = np.empty((slot0.shape[0], 2))
    out[:, 0] = alf[row - 1, col - 1, 0]
    out[:, 1] = alf[row - 1, col - 1, 1]
    if has_down:
        out[:, 0] += bet * (slot0 == 0)
        out[:, 1] += bet * (slot0 == 1)
    if has_right:
        out[:, 0] += bet * (slotR == 0)
        out[:, 1] += bet * (slotR == 1)
    if has_ne:
        out[:, 0] += gamma_ne * (slotNE == 0)
        out[:, 1] += gamma_ne * (slotNE == 1)
    if has_nw:
        out[:, 0] += gamma_nw * (slotNW == 0)
        out[:, 1] += gamma_nw * (slotNW == 1)
    return out


def _marginals_fast(alf, bet, gamma_ne, gamma_nw, obs_by_scan):
    n1, n2, d = alf.shape
    assert d == 2
    n = n1 * n2
    B = _full_buffer_length(n1, gamma_ne, gamma_nw)
    S = 1 << B
    slot0, slotR, slotNE, slotNW, parent = _state_maps(n1, B)

    # ---- Forward table F[i] over FULL-WIDTH states, faithful to reference ----
    # Reference indexes F[i] by a buffer of length ns[i]; in the bulk ns[i]=B.
    # We keep full-width S states throughout. In the bulk the recursion is
    #   F[i][xval] = sum_l q(l,xval,i) * F[i-1][parent[xval,l]].
    # In the last column the reference buffer shrinks; with full-width states the
    # shrink is equivalent to the off-lattice neighbours being absent, which is
    # already encoded by has_down/has_right/has_ne/has_nw in _logq_arr. The only
    # extra care: in the last column several full-width states map to the same
    # logical (shorter) buffer, so F carries redundant copies that stay consistent
    # under the same recursion. Validated against the reference path in the tests.
    F = [None] * (n + 1)
    logscale = np.zeros(n + 1)

    for i in range(1, n + 1):
        col = (i - 1) // n1 + 1
        row = (i - 1) % n1 + 1
        q = np.exp(_logq_arr(alf, bet, gamma_ne, gamma_nw,
                             row, col, n1, n2, slot0, slotR, slotNE, slotNW))  # (S,2)
        want = obs_by_scan.get(i)
        if want is not None:
            q[:, 1 - want] = 0.0
        if i == 1:
            Fi = q[:, 0] + q[:, 1]
        else:
            Fprev = F[i - 1]
            Fi = q[:, 0] * Fprev[parent[:, 0]] + q[:, 1] * Fprev[parent[:, 1]]
        sc = Fi.max()
        if not np.isfinite(sc) or sc <= 0:
            raise ValueError(f"forward mass invalid at step {i}.")
        F[i] = Fi / sc
        logscale[i] = logscale[i - 1] + np.log(sc)

    # logZ from the full-width forward sum overcounts: at the final site the
    # logical buffer is empty (ns[n]=0), so every logical completion is represented
    # by exactly 2^B equal-mass full-width copies. The overcount is therefore
    # exactly B*log(2) (B=n1 first order, n1+1 second order), verified against brute
    # force across geometries to machine precision. Subtract it to recover the true
    # logZ. (Self-contained and fast; avoids the slow reference forward table.)
    logZ = float(logscale[n] + np.log(F[n].sum()) - B * np.log(2.0))

    # ---- Backward distribution pass (dense), faithful to reference ----
    # D[s] = posterior prob the running buffer equals s. Conditional at site j:
    #   P(x_j=l|s) ∝ q(l,s,j) * F[j-1][parent[s,l]]   (and the j==1, j==n cases).
    marg = np.zeros((n1, n2, d))
    D = np.zeros(S)
    D[0] = 1.0

    for t in range(1, n + 1):
        j = n - t + 1
        col = (j - 1) // n1 + 1
        row = (j - 1) % n1 + 1
        q = np.exp(_logq_arr(alf, bet, gamma_ne, gamma_nw,
                             row, col, n1, n2, slot0, slotR, slotNE, slotNW))
        want = obs_by_scan.get(j)
        if want is not None:
            q[:, 1 - want] = 0.0

        if j == 1:
            num0 = q[:, 0]
            num1 = q[:, 1]
        else:
            Fprev = F[j - 1]
            num0 = q[:, 0] * Fprev[parent[:, 0]]
            num1 = q[:, 1] * Fprev[parent[:, 1]]
        denom = num0 + num1
        safe = denom > 0
        c0 = np.zeros(S); c1 = np.zeros(S)
        c0[safe] = num0[safe] / denom[safe]
        c1[safe] = num1[safe] / denom[safe]

        m0 = float((D * c0).sum())
        m1 = float((D * c1).sum())
        tot = m0 + m1
        marg[row - 1, col - 1, 0] = m0 / tot
        marg[row - 1, col - 1, 1] = m1 / tot

        # advance: new buffer = parent[s,l]
        Dn = np.zeros(S)
        np.add.at(Dn, parent[:, 0], D * c0)
        np.add.at(Dn, parent[:, 1], D * c1)
        sD = Dn.sum()
        if sD <= 0 or not np.isfinite(sD):
            raise ValueError(f"distribution mass invalid at site {j}.")
        D = Dn / sD

    return marg, logZ


# ======================================================================
# Public API
# ======================================================================
def marginals(alf, bet, gamma_ne=0.0, gamma_nw=0.0, observations=None,
              return_logZ=False, force_n1_smaller=True, method="fast"):
    """
    Exact per-site posterior marginals for the second-order anisotropic binary MRF.

    alf : (n1, n2, 2) unary log-potentials (same as getQ_2ndorder_aniso).
    bet : scalar axis-aligned coupling (down + right).
    gamma_ne : scalar NE/SW diagonal coupling.
    gamma_nw : scalar NW/SE diagonal coupling.
        gamma_ne = gamma_nw recovers the isotropic-diagonal model;
        gamma_ne = gamma_nw = 0 recovers first-order Ising/Potts.
    observations : dict {(r, c): label}, 0-based (r,c) in ORIGINAL orientation;
        None for prior marginals.
    return_logZ : also return logZ (log masked partition fn = log p(y) up to the
        prior normaliser; plain logZ with no observations).
    force_n1_smaller : transpose internally so the buffer axis is the short one.
        The gammas pass through the (r,c)->(c,r) swap unchanged: NE/SW stays NE/SW
        and NW/SE stays NW/SE (same argument as in MRF_functions_2ndorder_aniso).
    method : "fast" (vectorised, d=2) or "reference" (dict oracle).

    Returns marg (n1,n2,2) with marg[r,c,v]=P(x_{r,c}=v|y); optionally logZ.
    """
    alf = np.asarray(alf, dtype=float)
    if alf.ndim != 3:
        raise ValueError("alf must have shape (n1, n2, d).")
    gamma_ne = float(gamma_ne)
    gamma_nw = float(gamma_nw)
    orig_n1, orig_n2, d = alf.shape
    obs = dict(observations) if observations else {}
    swapped = False
    if force_n1_smaller and orig_n1 > orig_n2:
        alf = np.transpose(alf, (1, 0, 2))
        swapped = True
        obs = {(c, r): v for (r, c), v in obs.items()}
        # NE/SW <-> NE/SW and NW/SE <-> NW/SE under transpose, so gammas unchanged.
    n1, n2, d = alf.shape
    obs_scan = _obs_by_scan(obs, n1, n2)

    if method == "fast":
        if d != 2:
            raise ValueError("fast path is binary-only; use method='reference'.")
        marg, logZ = _marginals_fast(alf, bet, gamma_ne, gamma_nw, obs_scan)
    elif method == "reference":
        marg, logZ = _marginals_ref(alf, bet, gamma_ne, gamma_nw, obs_scan)
    else:
        raise ValueError("method must be 'fast' or 'reference'.")

    if swapped:
        marg = np.transpose(marg, (1, 0, 2))
    return (marg, logZ) if return_logZ else marg


def posterior_mean(alf, bet, gamma_ne=0.0, gamma_nw=0.0, observations=None,
                   label=1, force_n1_smaller=True, method="fast"):
    """(n1,n2) field of P(x=label|y); for binary label=1 this is E[x|y]."""
    return marginals(alf, bet, gamma_ne=gamma_ne, gamma_nw=gamma_nw,
                     observations=observations, return_logZ=False,
                     force_n1_smaller=force_n1_smaller, method=method)[..., label]


# ======================================================================
# Brute force + self-tests
# ======================================================================
def _bruteforce(alf, bet, gamma_ne=0.0, gamma_nw=0.0, observations=None):
    n1, n2, d = alf.shape
    n = n1 * n2
    obs = observations or {}
    marg = np.zeros((n1, n2, d)); Z = 0.0
    for code in range(d ** n):
        x = np.zeros((n1, n2), dtype=int); c = code
        for gi in range(1, n + 1):
            col = (gi - 1) // n1; row = (gi - 1) % n1
            x[row, col] = c % d; c //= d
        if any(x[r, cc] != v for (r, cc), v in obs.items()):
            continue
        w = 0.0
        for col in range(n2):
            for row in range(n1):
                l = x[row, col]; w += alf[row, col, l]
                if row < n1 - 1 and x[row + 1, col] == l:           # down
                    w += bet
                if col < n2 - 1 and x[row, col + 1] == l:           # right
                    w += bet
                if row > 0 and col < n2 - 1 and x[row - 1, col + 1] == l:   # NE/SW
                    w += gamma_ne
                if row < n1 - 1 and col < n2 - 1 and x[row + 1, col + 1] == l:  # NW/SE
                    w += gamma_nw
        w = np.exp(w); Z += w
        for col in range(n2):
            for row in range(n1):
                marg[row, col, x[row, col]] += w
    return marg / Z, float(np.log(Z))


def _selftest():
    import time
    rng = np.random.default_rng(0)

    def grids():
        # (alf, bet, gamma_ne, gamma_nw, obs)
        out = []
        a = np.zeros((4, 3, 2)); out.append((a, 0.7, 0.0, 0.0, None))               # 1st order
        a = np.zeros((4, 3, 2)); out.append((a, 0.5, 0.3, 0.3, None))               # isotropic diag
        a = np.zeros((4, 3, 2)); out.append((a, 0.5, 0.4, -0.2, None))              # anisotropic diag
        a = np.zeros((4, 3, 2))
        [a.__setitem__((slice(None), cc, 0), cc / 6 - 0.25) for cc in range(3)]
        out.append((a, 0.5, 0.3, 0.1, {(0, 0): 1, (3, 2): 0, (1, 1): 1}))          # obs + diag
        a = rng.normal(size=(4, 4, 2)) * 0.4
        out.append((a, 0.45, 0.25, 0.15, {(0, 1): 1, (2, 3): 0}))
        a = rng.normal(size=(3, 5, 2)) * 0.3
        out.append((a, 0.4, 0.2, 0.0, {(1, 2): 0}))                                 # ne only
        a = rng.normal(size=(3, 5, 2)) * 0.3
        out.append((a, 0.4, 0.0, 0.3, None))                                        # nw only
        return out

    print("=" * 72); print("A) fast vs reference vs brute force (with diagonals)")
    for k, (a, b, gne, gnw, o) in enumerate(grids()):
        mf, lzf = marginals(a, b, gamma_ne=gne, gamma_nw=gnw, observations=o,
                            return_logZ=True, force_n1_smaller=False, method="fast")
        mr, lzr = marginals(a, b, gamma_ne=gne, gamma_nw=gnw, observations=o,
                            return_logZ=True, force_n1_smaller=False, method="reference")
        mb, lzb = _bruteforce(a, b, gamma_ne=gne, gamma_nw=gnw, observations=o)
        print(f"  grid {k} (gne={gne:+.2f}, gnw={gnw:+.2f}): "
              f"fast-bf={np.abs(mf - mb).max():.2e}  "
              f"ref-bf={np.abs(mr - mb).max():.2e}  "
              f"logZ fast-bf={abs(lzf - lzb):.2e} ref-bf={abs(lzr - lzb):.2e}")

    print("=" * 72); print("B) transpose invariance (fast), 3x5 anisotropic")
    a = rng.normal(size=(3, 5, 2)) * 0.3
    ma = marginals(a, 0.5, gamma_ne=0.3, gamma_nw=0.1, force_n1_smaller=False)
    mb = marginals(a, 0.5, gamma_ne=0.3, gamma_nw=0.1, force_n1_smaller=True)
    print(f"  max|A-B| = {np.abs(ma - mb).max():.2e}")

    print("=" * 72); print("C) logZ vs getQ_2ndorder_aniso.get_q, 6x4 anisotropic")
    a = rng.normal(size=(6, 4, 2)) * 0.2
    _, lzg = getQ.get_q(a, 0.6, gamma_ne=0.25, gamma_nw=0.1, rng=rng, return_logZ=True)
    _, lzf = marginals(a, 0.6, gamma_ne=0.25, gamma_nw=0.1,
                       return_logZ=True, force_n1_smaller=False)
    print(f"  diff = {abs(lzg - lzf):.2e}")

    print("=" * 72); print("D) MC sample-mean sanity, 5x4 anisotropic")
    a = np.zeros((5, 4, 2))
    me = posterior_mean(a, 0.6, gamma_ne=0.3, gamma_nw=0.2, force_n1_smaller=False)
    S = 20000; acc = np.zeros((5, 4))
    sampler, _ = __import__("MRF_functions_2ndorder_aniso").build_sampler(
        a, 0.6, gamma_ne=0.3, gamma_nw=0.2)
    for _ in range(S):
        acc += sampler(rng)
    print(f"  max|exact-MC({S})| = {np.abs(me - acc / S).max():.2e} (~{1/np.sqrt(S):.2e})")

    print("=" * 72); print("E) SPEED at 12x24 anisotropic")
    n1, n2 = 12, 24
    a = np.zeros((n1, n2, 2))
    xo = getQ.get_q(a, 0.7, gamma_ne=0.2, gamma_nw=0.1, rng=rng)
    obs = {(r, c): int(xo[r, c]) for c in [6, 18] for r in range(n1)}
    t = time.time()
    m, lz = marginals(a, 0.7, gamma_ne=0.2, gamma_nw=0.1,
                      observations=obs, return_logZ=True)
    print(f"  two wells: {time.time() - t:.3f}s  "
          f"obs pinned={np.allclose(m[:, [6, 18], 1], xo[:, [6, 18]])}")


if __name__ == "__main__":
    _selftest()
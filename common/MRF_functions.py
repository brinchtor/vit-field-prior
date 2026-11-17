# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 14:57:24 2026

@author: brinc
"""

import numpy as np

# Import your module (file name: getQ.py)
import getQ


def build_sampler(alf: np.ndarray, bet: float):
    """
    Precompute the forward DP once, then return a fast sampler closure.
    alf: (n1, n2, d) unary log-potentials
    bet: scalar
    Returns: (sample_fn, logZ)
      - sample_fn(rng) -> q of shape (n1, n2) with labels in {0..d-1}
      - logZ is the exact log partition function for this model
    """
    n1, n2, d = alf.shape
    n = n1 * n2
    num0 = d ** n1

    ns = np.zeros(n + 1, dtype=int)
    num = np.zeros(n + 1, dtype=int)
    z_list = [None] * (n + 1)
    
    # ---------- Forward recursion (copied from getQ.get_q, but separated) ----------
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
                xval = getQ.index2vec(k, d, ns[i])
                acc = 0.0
                for l in range(d):
                    if i > 1:
                        prev_ns = ns[i - 1]
                        prev_vec = np.concatenate(([l], xval[:prev_ns - 1]))
                        ind = getQ.vec2index(prev_vec, d)
                        acc += getQ.qeval(l, xval, i, bet, alf) * z_list[i - 1][ind - 1]
                    else:
                        acc += getQ.qeval(l, xval, i, bet, alf)
                z_i[k - 1] = acc
            z_list[i] = z_i
        else:
            # i == n
            z_i = np.zeros(1, dtype=float)
            acc = 0.0
            for l in range(d):
                acc += np.exp(alf[n1 - 1, n2 - 1, l]) * z_list[i - 1][l]
            z_i[0] = acc
            z_list[i] = z_i
    
    Z = z_list[n][0]
    if not np.isfinite(Z) or Z <= 0.0:
        raise ValueError("Partition function is zero/invalid: evidence may be impossible or underflow occurred.")
    logZ = float(np.log(z_list[n][0]))

    # ---------- Backward sampling closure ----------
    def sample(rng: np.random.Generator) -> np.ndarray:
        x = np.zeros(n, dtype=int)
        probvec = np.zeros(d, dtype=float)
        buff = None

        for t in range(1, n + 1):
            j = n - t + 1  # 1-based

            if j == n:
                denom = z_list[n][0]
                for l in range(d):
                    probvec[l] = (np.exp(alf[n1 - 1, n2 - 1, l]) * z_list[j - 1][l]) / denom
                x[j - 1] = getQ.drawval(probvec, rng)
            else:
                buffp = buff.copy()
                indbuff = getQ.vec2index(buff, d)

                for l in range(d):
                    if j > n - n1:
                        buffi = np.concatenate(([l], buff[:ns[j]]))
                    else:
                        buffi = np.concatenate(([l], buff[:ns[j] - 1]))

                    indbuffi = getQ.vec2index(buffi, d)

                    if j == 1:
                        probvec[l] = getQ.qeval(l, buff, j, bet, alf) / z_list[j][indbuff - 1]
                    else:
                        probvec[l] = (
                            getQ.qeval(l, buff, j, bet, alf)
                            * z_list[j - 1][indbuffi - 1]
                            / z_list[j][indbuff - 1]
                        )

                x[j - 1] = getQ.drawval(probvec, rng)

            # update buffer
            if j == n:
                buff = np.array([x[j - 1]], dtype=int)
            elif j > n - n1:
                buff = np.concatenate(([x[j - 1]], buffp))
            else:
                buff = np.concatenate(([x[j - 1]], buffp[:n1 - 1]))

        return getQ.vec2mat(x, n1)

    return sample, logZ

def generate_samples(
    alf: np.ndarray,
    bet: float,
    N: int,
    seed: int = 0,
    force_n1_smaller: bool = True,
    return_all: bool = False,
    dtype=np.uint16,
):
    """
    Generates N samples (each n1 x n2) and saves to a compressed npz.

    Saves:
      - samples: (N, n1, n2)
      - bet, d, logZ, swapped, original_shape
    """
    alf = np.asarray(alf)
    if alf.ndim != 3:
        raise ValueError("alf must have shape (n1, n2, d).")

    orig_shape = alf.shape
    swapped = False

    # Make n1 the smaller dimension (faster DP) by swapping axes if desired.
    if force_n1_smaller and alf.shape[0] > alf.shape[1]:
        alf = np.transpose(alf, (1, 0, 2))
        swapped = True

    n1, n2, d = alf.shape
    sampler, logZ = build_sampler(alf, bet)

    rng = np.random.default_rng(seed)
    samples = np.empty((N, n1, n2), dtype=dtype)

    for k in range(N):
        q = sampler(rng)  # (n1, n2), labels 0..d-1
        samples[k] = q.astype(dtype, copy=False)

    # If we swapped to make sampling fast, swap back so saved shape matches original alf's (n1,n2)
    if swapped:
        samples = np.transpose(samples, (0, 2, 1))  # (N, orig_n1, orig_n2)
    
    if return_all:
        return samples, d, logZ, swapped, orig_shape
    else:
        return samples

def generate_samples_npz(
    alf: np.ndarray,
    bet: float,
    N: int,
    out_path: str,
    seed: int = 0,
    force_n1_smaller: bool = True,
    dtype=np.uint16,
):
    """
    Generates N samples (each n1 x n2) and saves to a compressed npz.

    Saves:
      - samples: (N, n1, n2)
      - bet, d, logZ, swapped, original_shape
    """
    samples, d, logZ, swapped, orig_shape = generate_samples(alf, bet, N, seed, force_n1_smaller=force_n1_smaller, return_all=True, dtype=dtype,
    )
    
    rng = np.random.default_rng(seed)
    total_samples = np.concatenate([samples, 1 - samples], axis=0)
    shuffled_idx = rng.permutation(len(total_samples))
    total_samples = total_samples[shuffled_idx]

    np.savez_compressed(
        out_path,
        images=total_samples,
        bet=float(bet),
        d=int(d),
        logZ=float(logZ),
        swapped=swapped,
        original_shape=np.array(orig_shape, dtype=int),
        seed=int(seed),
    )


# ---------------- Example usage ----------------
if __name__ == "__main__":
    # Example: alf is (n1, n2, d)
    # alf = np.load("alf.npy")  # you provide this
    # bet = 0.7
    # N = 5000

    # generate_samples_npz(alf, bet, N, "potts_samples.npz", seed=123, force_n1_smaller=True)
    pass

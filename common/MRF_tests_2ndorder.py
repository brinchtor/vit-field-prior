"""
Sanity-check the second-order MRF sampler against brute-force enumeration
on a small grid where exhaustive summation is feasible.
"""
import itertools
import numpy as np

import getQ_2ndorder as getQ
import MRF_functions_2ndorder as mrf


def brute_force_logZ_and_marginals(alf, bet, gamma):
    """
    Enumerate all 2^(n1*n2) binary configurations and compute logZ and
    per-site marginals P(x_{r,c} = 1) directly from the energy.

    Energy follows the column-major scan-order convention used by qeval:
      energy(x) = sum_i [ alf[r,c,x_i]
                          + bet * 1{x_i = x_down}   (if exists)
                          + bet * 1{x_i = x_right}  (if exists)
                          + gamma * 1{x_i = x_NE}   (if exists)
                          + gamma * 1{x_i = x_SE}   (if exists) ]
    """
    n1, n2, d = alf.shape
    n = n1 * n2

    log_unnorm = []
    configs = []
    for bits in itertools.product(range(d), repeat=n):
        x = np.array(bits, dtype=int).reshape((n1, n2), order='F')
        e = 0.0
        for col in range(1, n2 + 1):
            for row in range(1, n1 + 1):
                xi = x[row - 1, col - 1]
                e += alf[row - 1, col - 1, xi]
                # down
                if row < n1:
                    if xi == x[row, col - 1]:
                        e += bet
                # right
                if col < n2:
                    if xi == x[row - 1, col]:
                        e += bet
                # NE-diagonal: site (row-1, col+1), exists iff row > 1 and col < n2
                if row > 1 and col < n2:
                    if xi == x[row - 2, col]:
                        e += gamma
                # SE-diagonal: site (row+1, col+1), exists iff row < n1 and col < n2
                if row < n1 and col < n2:
                    if xi == x[row, col]:
                        e += gamma
        log_unnorm.append(e)
        configs.append(x)

    log_unnorm = np.array(log_unnorm)
    m = log_unnorm.max()
    Z = np.exp(log_unnorm - m).sum()
    logZ = m + np.log(Z)

    # Per-site marginals for label == 1 (assuming d == 2)
    probs = np.exp(log_unnorm - logZ)
    marg = np.zeros((n1, n2))
    for x, p in zip(configs, probs):
        marg += p * (x == 1)
    return logZ, marg


def run_test(n1, n2, bet, gamma, seed=0, N=20000):
    np.random.seed(seed)
    d = 2
    # Random unary log-potentials so the marginals are non-trivial
    rng_alf = np.random.default_rng(seed)
    alf = rng_alf.normal(scale=0.3, size=(n1, n2, d))

    # Brute force
    bf_logZ, bf_marg = brute_force_logZ_and_marginals(alf, bet, gamma)

    # DP logZ
    _, _, _, dp_logZ = getQ.forward_dp(alf, bet, gamma)

    # Empirical marginals from N samples
    sampler, logZ_again = mrf.build_sampler(alf, bet, gamma)
    rng = np.random.default_rng(seed + 1)
    counts = np.zeros((n1, n2))
    for _ in range(N):
        q = sampler(rng)
        counts += (q == 1)
    emp_marg = counts / N

    return {
        'n1': n1, 'n2': n2, 'bet': bet, 'gamma': gamma,
        'bf_logZ': bf_logZ, 'dp_logZ': dp_logZ,
        'logZ_diff': dp_logZ - bf_logZ,
        'bf_marg': bf_marg, 'emp_marg': emp_marg,
        'max_abs_marg_diff': np.abs(bf_marg - emp_marg).max(),
        'expected_mc_se': 1.0 / (2 * np.sqrt(N)),  # rough scale of |emp - true|
    }


if __name__ == "__main__":
    print("="*70)
    print("Test 1: first-order Ising (gamma = 0), 3x3 grid, bet = 0.7")
    print("="*70)
    r = run_test(n1=3, n2=3, bet=0.7, gamma=0.0, seed=0, N=20000)
    print(f"  brute-force logZ = {r['bf_logZ']:.10f}")
    print(f"  DP logZ          = {r['dp_logZ']:.10f}")
    print(f"  difference       = {r['logZ_diff']:.3e}")
    print(f"  max |emp - true| marginal diff = {r['max_abs_marg_diff']:.4f}")
    print(f"  rough MC noise scale           = {r['expected_mc_se']:.4f}")

    print("\n" + "="*70)
    print("Test 2: second-order (gamma = 0.4), 3x3 grid, bet = 0.5")
    print("="*70)
    r = run_test(n1=3, n2=3, bet=0.5, gamma=0.4, seed=1, N=20000)
    print(f"  brute-force logZ = {r['bf_logZ']:.10f}")
    print(f"  DP logZ          = {r['dp_logZ']:.10f}")
    print(f"  difference       = {r['logZ_diff']:.3e}")
    print(f"  max |emp - true| marginal diff = {r['max_abs_marg_diff']:.4f}")
    print(f"  rough MC noise scale           = {r['expected_mc_se']:.4f}")
    print("\n  brute-force marginals:")
    print(r['bf_marg'])
    print("  empirical marginals:")
    print(r['emp_marg'])

    print("\n" + "="*70)
    print("Test 3: second-order, 3x4 non-square grid, bet=0.3, gamma=0.5")
    print("="*70)
    r = run_test(n1=3, n2=4, bet=0.3, gamma=0.5, seed=2, N=20000)
    print(f"  brute-force logZ = {r['bf_logZ']:.10f}")
    print(f"  DP logZ          = {r['dp_logZ']:.10f}")
    print(f"  difference       = {r['logZ_diff']:.3e}")
    print(f"  max |emp - true| marginal diff = {r['max_abs_marg_diff']:.4f}")
    print(f"  rough MC noise scale           = {r['expected_mc_se']:.4f}")

    print("\n" + "="*70)
    print("Test 4: second-order, 4x3 transposed-shape grid, bet=0.3, gamma=0.5")
    print("="*70)
    r = run_test(n1=4, n2=3, bet=0.3, gamma=0.5, seed=2, N=20000)
    print(f"  brute-force logZ = {r['bf_logZ']:.10f}")
    print(f"  DP logZ          = {r['dp_logZ']:.10f}")
    print(f"  difference       = {r['logZ_diff']:.3e}")
    print(f"  max |emp - true| marginal diff = {r['max_abs_marg_diff']:.4f}")
    print(f"  rough MC noise scale           = {r['expected_mc_se']:.4f}")
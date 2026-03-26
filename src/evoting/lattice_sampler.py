"""
Lattice Sampler — SampleD (Micciancio-Peikert, EUROCRYPT 2012)

Implements the trapdoor-based preimage sampling algorithm used in:
  Jeudy, Roux-Langlois, Sanders — "Lattice Signature with Efficient Protocols,
  Application to Anonymous Credentials" (Crypto 2023)

Key fix over previous version: proper gadget decomposition in base 2.
The old code set z[:n] = t' mod q (values up to q/2 ≈ 1600), making
signatures fail verification. The correct G-sampling decomposes each 
component of t' in base 2, producing z ∈ {0,1}^m2.
"""

import numpy as np
from math import ceil, log2


def discrete_gaussian_sample(dim: int, sigma: float) -> np.ndarray:
    """
    Sample from a discrete Gaussian D_{Z^dim, sigma}.
    Approximated by rounding continuous Gaussian samples.
    
    In production, use constant-time samplers (cf. truncated-sampler C code,
    which uses Klein sampling on the gadget lattice + rejection sampling).
    """
    return np.round(np.random.normal(0, sigma, size=(dim, 1))).astype(np.int64)


def gadget_solve(t_val: int, q: int, base: int = 2) -> np.ndarray:
    """
    Solve g^T * z = t mod q for a SHORT z, where g = [1, base, base^2, ...].
    
    This is the core of G-sampling: decomposition of t mod q in the given base,
    producing coefficients in {0, ..., base-1}. This is what makes the output
    SHORT — O(1) per component instead of O(q).
    
    Reference: [MP12] Section 4 (Gadget trapdoor)
    """
    k = ceil(log2(q) / log2(base)) if base > 1 else int(q)
    z = np.zeros(k, dtype=np.int64)
    val = int(t_val) % q
    for i in range(k):
        z[i] = val % base
        val = val // base
    return z


def sampleD(R: np.ndarray, A: np.ndarray, tau: int, target: np.ndarray,
            q: int, G: np.ndarray, sigma: float = 2000, sigma1: float = 100000) -> np.ndarray:
    """
    SampleD algorithm (Micciancio-Peikert, EUROCRYPT 2012).

    Given the trapdoor R (where B = A*R mod q), finds a short vector v such that:
        [A | tau*G - B] * v = target mod q

    Steps:
        1. Perturbation:    p <- D_{Z^m1, sigma1}
        2. Syndrome:        t' = tau^{-1} * (target - A*p) mod q
        3. G-sampling:      solve G*z = t' mod q for SHORT z (binary decomposition)
        4. Reconstruction:  v = [p + R*z; z]

    The gadget matrix G = I_n ⊗ g with g = [1, 2, 4, ..., 2^{k-1}].
    G-sampling decomposes each component of t' in base 2 → coefficients in {0,1}.

    Correctness proof (inline):
        A_tau * v = A*(p + R*z) + (tau*G - B)*z
                  = A*p + A*R*z + tau*G*z - B*z
                  = A*p + B*z + tau*G*z - B*z       [since A*R = B]
                  = A*p + tau*G*z
                  = A*p + tau * tau^{-1} * (target - A*p)
                  = target mod q                     QED

    Reference: [MP12] Thm 5.1, paper Section 3.1, src/algorithm.md
    """
    m1, m2 = R.shape
    n = A.shape[0]
    k = m2 // n  # number of gadget digits per dimension

    # Step 1: Perturbation sampling (hides the trapdoor)
    p = discrete_gaussian_sample(m1, sigma1)

    # Step 2: Compute syndrome for gadget inversion
    residue = (target.flatten() - (A @ p).flatten()) % q

    # tau^{-1} mod q via Fermat's little theorem (q prime)
    if tau == 0:
        raise ValueError("Tag tau must be non-zero (tau ∈ T = Z_q' \\ {0})")
    tau_inv = pow(int(tau), q - 2, q)
    t_prime = (tau_inv * residue) % q

    # Step 3: G-sampling — solve G*z = t' mod q with SHORT z
    # G = I_n ⊗ [1, 2, 4, ..., 2^{k-1}]
    # For each of the n components of t', decompose in base 2
    z = np.zeros((m2, 1), dtype=np.int64)
    for i in range(n):
        decomp = gadget_solve(int(t_prime[i]), q, base=2)
        for j in range(min(k, len(decomp))):
            z[i * k + j, 0] = decomp[j]

    # Step 4: Reconstruct preimage v = [p + R*z; z]
    v_top = (p + R @ z).astype(np.int64)
    v = np.vstack([v_top, z]).astype(np.int64)

    return v

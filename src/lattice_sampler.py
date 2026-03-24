import numpy as np
from typing import Tuple


def sample_discrete_gaussian_1d(center: float, sigma: float) -> int:
    if sigma < 1e-10:
        return int(round(center))

    tail_cut = 12  # number of standard deviations for the support
    lower = int(np.floor(center - tail_cut * sigma))
    upper = int(np.ceil(center + tail_cut * sigma))

    while True:
        z = np.random.randint(lower, upper + 1)
        # Probability proportional to exp(-|z - center|^2 / (2 * sigma^2))
        prob = np.exp(-((z - center) ** 2) / (2.0 * sigma ** 2))
        if np.random.random() < prob:
            return z


def sample_discrete_gaussian_vector(center: np.ndarray, sigma: float) -> np.ndarray:
    """
    Samples a vector from a spherical discrete Gaussian
    (each coordinate is independent).

    Parameters
    ----------
    center : np.ndarray
        Center vector (shape (d,1) or (d,))
    sigma : float
        Width parameter

    Returns
    -------
    np.ndarray
        Sampled vector, shape identical to center
    """
    flat = center.flatten()
    result = np.array([sample_discrete_gaussian_1d(float(c), sigma) for c in flat],
                      dtype=np.int64)
    return result.reshape(center.shape)


def sample_g_coset(target_val: int, q: int, sigma: float) -> int:
    """
    Solves 1 * z = target_val mod q for a component of the gadget matrix
    G = I_n. Finds a short z such that z = target_val (mod q).

    We sample z = target_mod + k*q with k integer, choosing
    k such that z is short (centered around 0).

    Parameters
    ----------
    target_val : int
        Target value modulo q
    q : int
        Modulus
    sigma : float
        Gaussian parameter

    Returns
    -------
    int
        short z such that z = target_val (mod q)
    """
    target_mod = int(target_val) % q

    # We want z = target_mod + k*q small
    # => k ~ -target_mod / q
    k_center = -target_mod / q

    # Sigma for k: if we want |z| ~ sigma, then |k| ~ sigma/q
    sigma_k = max(sigma / q, 1.0)

    k = sample_discrete_gaussian_1d(k_center, sigma_k)
    z = target_mod + k * q
    return int(z)


def sample_g_vector(target: np.ndarray, n: int, m2: int, q: int, sigma: float) -> np.ndarray:
    """
    Solves G*z = target mod q for G = I_n extended to n x m2.

    G is of the form [I_n | 0_{n x (m2-n)}].
    So G*z = z[0:n]. We need z[0:n] = target (mod q),
    and z[n:m2] is free (sampled from D_{Z, sigma, 0}).

    Parameters
    ----------
    target : np.ndarray
        Target vector, shape (n, 1)
    n : int
        Number of rows of G
    m2 : int
        Number of columns of G (number of components of z)
    q : int
        Modulus
    sigma : float
        Gaussian parameter

    Returns
    -------
    np.ndarray
        Vector z of shape (m2, 1) such that G*z = target (mod q) and short z
    """
    z = np.zeros((m2, 1), dtype=np.int64)

    # First n components: z_i = target_i mod q (coset sampling)
    for i in range(n):
        z[i, 0] = sample_g_coset(target[i, 0], q, sigma)

    # The remaining m2-n components: free, sampled from D_{Z, sigma, 0}
    for i in range(n, m2):
        z[i, 0] = sample_discrete_gaussian_1d(0.0, sigma)

    return z


def sampleD(R: np.ndarray, A: np.ndarray, tau: int, target: np.ndarray,
            q: int, G: np.ndarray, sigma: float = 2000.0, sigma1: float = 100000.0) -> np.ndarray:
    """
    SampleD algorithm (Micciancio-Peikert 2012) - Pre-image sampling with trapdoor.

    Given:
      - A in Z_q^{n x m1}           (public matrix)
      - R in Z^{m1 x m2}            (trapdoor: A*R = B mod q)
      - G in Z_q^{n x m2}           (gadget matrix, here padded I_n)
      - tau in Z_q                  (tag, must be invertible mod q)
      - target in Z_q^{n x 1}       (target vector)

    Finds v = [v1; v2] in Z^{(m1+m2) x 1} such that:
      [A | tau*G - B] * v = target mod q
      with short v (bounded norm).

    Scheme:
      F = [R; I_{m2}]  =>  A_tau * F = tau * G
      1. Sample p in Z^{m1} (Gaussian perturbation, sigma1)
      2. Compute target' = tau^{-1} * (target - A*p) mod q
      3. G-sample z such that G*z = target' mod q (short z)
      4. v = [p + R*z; z]

    Parameters
    ----------
    R : np.ndarray
        Trapdoor, shape (m1, m2)
    A : np.ndarray
        Public matrix, shape (n, m1)
    tau : int
        Tag (scalar, must be invertible mod q)
    target : np.ndarray
        Target vector, shape (n, 1)
    q : int
        Modulus
    G : np.ndarray
        Gadget matrix, shape (n, m2)
    sigma : float
        Gaussian parameter for G-sampling (controls size of v2)
    sigma1 : float
        Gaussian parameter for perturbation (controls size of v1)

    Returns
    -------
    np.ndarray
        Vector v of shape (m1+m2, 1) such that A_tau*v = target mod q
    """
    m1, m2 = R.shape
    n = A.shape[0]

    # ============================================================
    # Step 1: Perturbation sampling
    # ============================================================
    # Sample p in Z^{m1} from D_{Z^m1, sigma1, 0}
    p = sample_discrete_gaussian_vector(np.zeros((m1, 1)), sigma1)

    # ============================================================
    # Step 2: Remainder calculation for G-sampling
    # ============================================================
    # target' = tau^{-1} * (target - A*p) mod q
    Ap = np.dot(A.astype(np.int64), p.astype(np.int64)) % q
    residual = (target.astype(np.int64) - Ap.astype(np.int64)) % q

    # Calculate tau^{-1} mod q
    tau_inv = pow(int(tau), -1, q)

    target_for_g = (tau_inv * residual.astype(np.int64)) % q

    # ============================================================
    # Step 3: G-sampling
    # ============================================================
    # Solve G*z = target_for_g mod q with short z
    # G = [I_n | 0], so z[0:n] must satisfy the constraint,
    # and z[n:m2] is free
    z = sample_g_vector(target_for_g, n, m2, q, sigma)

    # ============================================================
    # Step 4: Combination via trapdoor
    # ============================================================
    # v = [p + R*z; z]
    # A_tau * v = A*(p + R*z) + (tau*G - B)*z
    #           = A*p + A*R*z + tau*G*z - B*z
    #           = A*p + B*z + tau*G*z - B*z       [since A*R = B]
    #           = A*p + tau*G*z
    #           = A*p + tau * target_for_g          [since G*z = target_for_g]
    #           = A*p + tau * tau^{-1} * (target - A*p)
    #           = A*p + target - A*p
    #           = target  mod q                     QED
    Rz = np.dot(R.astype(np.int64), z.astype(np.int64))
    v1 = (p.astype(np.int64) + Rz)  # shape (m1, 1)
    v2 = z                            # shape (m2, 1)

    v = np.vstack([v1, v2]).astype(np.int64)
    return v


def verify_preimage(A: np.ndarray, R: np.ndarray, G: np.ndarray,
                    tau: int, target: np.ndarray, v: np.ndarray, q: int) -> bool:
    """
    Verifies that the vector v satisfies A_tau * v = target mod q.

    Parameters
    ----------
    A : np.ndarray
        Public matrix, shape (n, m1)
    R : np.ndarray
        Trapdoor, shape (m1, m2)
    G : np.ndarray
        Gadget matrix, shape (n, m2)
    tau : int
        Tag
    target : np.ndarray
        Target vector, shape (n, 1)
    v : np.ndarray
        Pre-image vector, shape (m1+m2, 1)
    q : int
        Modulus

    Returns
    -------
    bool
        True if A_tau * v = target mod q
    """
    n, m1 = A.shape
    m2 = G.shape[1]
    B = np.dot(A.astype(np.int64), R.astype(np.int64)) % q

    A_tau_right = (tau * G.astype(np.int64) - B.astype(np.int64)) % q
    A_tau = np.hstack([A.astype(np.int64), A_tau_right])

    result = np.dot(A_tau, v.astype(np.int64)) % q
    target_mod = target.astype(np.int64) % q

    return np.array_equal(result % q, target_mod % q)

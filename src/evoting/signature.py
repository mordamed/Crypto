import numpy as np
from math import ceil, log2, sqrt
from typing import Tuple

from .lattice_sampler import sampleD, discrete_gaussian_sample

# Lattice-Based Signature Scheme with Efficient Protocols

class LatticeSignatureScheme:
    def __init__(self, n=64, q=3329, m1=128, m2=None, m3=32):
        """
        Algorithm 3.1: Setup

        Parameters:
            n  - SIS dimension (ring degree in module setting)
            q  - modulus (prime)
            m1 - commitment randomness dimension
            m2 - trapdoor dimension (= n * ceil(log2(q)) if None)
            m3 - maximum bit-size of message
        """
        self.n = n
        self.q = q
        self.m1 = m1
        self.k = ceil(log2(q))
        self.m2 = n * self.k if m2 is None else m2
        self.m3 = m3
        self.m = m1 + self.m2

        # Public commitment matrix D <-$ U(Z_q^{n x m3})
        self.D = np.random.randint(0, q, size=(n, m3), dtype=np.int64)

        # Gadget matrix G = I_n kron g where g = [1, 2, 4, ..., 2^{k-1}]
        g = np.array([2**i for i in range(self.k)], dtype=np.int64).reshape(1, -1)
        self.G = np.kron(np.eye(n, dtype=np.int64), g)

        # Gaussian widths - Algorithm 3.1, steps 13-15
        self.sigma = 2000         # preimage sampling width sigma (Lemma 2.6)
        self.sigma2 = 500         # commitment randomness width sigma2
        self.sigma1 = int(sqrt(self.sigma**2 + self.sigma2**2))  # sigma1 = sqrt(sigma^2 + sigma2^2)

        # Verification bounds - Algorithm 3.4, step 3
        # ||v1||_inf <= sigma1 * log2(m1),  ||v2||_inf <= sigma * log2(m2)
        self.bound_v1 = self.sigma1 * log2(m1) if m1 > 1 else self.sigma1
        self.bound_v2 = self.sigma * log2(self.m2) if self.m2 > 1 else self.sigma

    def keygen(self) -> Tuple[dict, dict]:
        """
        Algorithm 3.2: KeyGen

        Returns:
            pk = {A, B, u}  where B = A*R mod q
            sk = {R}         ternary trapdoor matrix
        """
        A = np.random.randint(0, self.q, size=(self.n, self.m1), dtype=np.int64)
        R = np.random.randint(-1, 2, size=(self.m1, self.m2), dtype=np.int64)
        B = (A @ R) % self.q
        u = np.random.randint(0, self.q, size=(self.n, 1), dtype=np.int64)

        pk = {'A': A, 'B': B, 'u': u}
        sk = {'R': R}
        return pk, sk

    def sign(self, sk: dict, pk: dict, m: np.ndarray, tag: int) -> dict:
        """
        Algorithm 3.3: Sign

        The paper's key innovation in 3 steps:
        1. Commit: c = A*r + D*m mod q
        2. SampleD: find v' s.t. [A | tau*G - B]*v' = u + c mod q
        3. Absorb: v = v' - [r; 0]  <- removes r from the signature

        This absorption is what eliminates the commitment opening from the
        signature, saving thousands of ZK proof witnesses.
        """
        A, B, u = pk['A'], pk['B'], pk['u']
        R = sk['R']

        r = discrete_gaussian_sample(self.m1, self.sigma2)
        c = (A @ r + self.D @ m) % self.q
        target = (u + c) % self.q

        v_prime = sampleD(R, A, tag, target, self.q, self.G, sigma=self.sigma)

        r_padded = np.vstack([r, np.zeros((self.m2, 1), dtype=np.int64)])
        v = v_prime - r_padded

        return {'tau': tag, 'v': v}

    def verify(self, pk: dict, m: np.ndarray, sig: dict) -> bool:
        """
        Algorithm 3.4: Verify

        Checks:
        1. [A | tau*G - B] * v = u + D*m  (mod q)       algebraic correctness
        2. ||v1||_inf <= sigma1 * log2(m1)               shortness of v1
        3. ||v2||_inf <= sigma * log2(m2)                shortness of v2
        4. tau in T = Z_q \\ {0}                         valid tag
        """
        A, B, u = pk['A'], pk['B'], pk['u']
        tau, v = sig['tau'], sig['v']

        v1 = v[:self.m1]
        v2 = v[self.m1:]

        A_tau_right = (tau * self.G - B) % self.q
        A_tau = np.hstack([A, A_tau_right])

        lhs = (A_tau.astype(np.int64) @ v.astype(np.int64)) % self.q
        rhs = (u + self.D @ m) % self.q

        # Algorithm 3.4, step 3: tau in T = Z_q \ {0}
        tau_ok = (0 < tau % self.q < self.q)
        equation_ok = np.array_equal(lhs % self.q, rhs % self.q)
        v1_ok = np.max(np.abs(v1)) <= self.bound_v1
        v2_ok = np.max(np.abs(v2)) <= self.bound_v2

        return tau_ok and equation_ok and v1_ok and v2_ok

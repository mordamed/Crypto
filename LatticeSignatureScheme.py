"""
Lattice Signature Scheme - Implementation complete avec SampleD reel
====================================================================

Basee sur l'algorithme de Micciancio-Peikert (2012) pour le pre-image
sampling avec trapdoor sur les reseaux euclidiens.

Ce fichier remplace le notebook LatticeSignatureScheme.ipynb avec une
implementation reelle de SampleD au lieu du mock.
"""

import numpy as np
from lattice_sampler import sampleD


class LatticeSignatureScheme:
    def __init__(self, n=64, q=3329, m1=128, m2=128, m3=32):
        """
        Algorithm 3.1 : Setup

        Parametres reduits par defaut (n=64) pour un temps d'execution
        raisonnable en demo. Pour la securite reelle, utiliser n=512+.
        """
        self.n = n
        self.q = q
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

        # Matrice D aleatoire publique pour cacher/engager le message
        self.D = np.random.randint(0, q, size=(n, m3))

        # Matrice Gadget G = [I_n | 0] de taille n x m2
        self.G = np.eye(n, m2)

        # Parametres gaussiens pour le sampling
        self.sigma = 2000        # parametre pour le G-sampling
        self.sigma1 = 100000     # parametre pour la perturbation

        # Bornes de verification (norme infinie)
        # v1 = p + R*z : |p| ~ sigma1, |R*z| ~ m2 * sigma (car R in {-1,0,1})
        # v2 = z : |z| ~ sigma
        self.sigma1_bound = 12 * self.sigma1 + self.m2 * 12 * self.sigma
        self.sigma_bound = 12 * self.sigma

    def keygen(self):
        """
        Algorithm 3.2 : Keygen
        Generates the public and secret keys.

        - A : matrice publique aleatoire (n x m1)
        - R : trapdoor (m1 x m2), entries in {-1, 0, 1}
        - B = A*R mod q : matrice derivee
        - u : vecteur public aleatoire (n x 1)
        """
        A = np.random.randint(0, self.q, size=(self.n, self.m1))
        R = np.random.randint(-1, 2, size=(self.m1, self.m2))
        B = np.dot(A, R) % self.q
        u = np.random.randint(0, self.q, size=(self.n, 1))

        pk = (A, B, u)
        sk = R
        return pk, sk

    def sign(self, sk, pk, message, state_tag):
        """
        Algorithm 3.3 : Sign
        Uses the real SampleD (Micciancio-Peikert 2012) for preimage sampling.

        Steps:
        1. Sample randomness r
        2. Compute commitment c = A*r + D*m mod q
        3. Use SampleD to find v' such that A_tau * v' = u + c mod q
        4. Return v = v' - [r; 0]
        """
        A, B, u = pk
        R = sk

        # r : Randomness
        r = np.random.randint(-2, 3, size=(self.m1, 1))

        # c : Commitment to m
        c = (np.dot(A, r) + np.dot(self.D, message)) % self.q

        # tau : Tag
        tau = state_tag

        # Pre-image sampling via SampleD (Micciancio-Peikert)
        target = (u + c) % self.q
        v_prime = sampleD(R, A, tau, target, self.q, self.G,
                          sigma=self.sigma, sigma1=self.sigma1)

        r_padded = np.vstack([r, np.zeros((self.m2, 1))])
        v = v_prime - r_padded

        return tau, v

    def verify(self, pk, message, signature):
        """
        Algorithm 3.4 : Verify
        Verify a signature using the public key.

        Checks:
        1. A_tau * v == u + D*m mod q  (equation correctness)
        2. ||v1||_inf <= sigma1_bound  (v1 is short)
        3. ||v2||_inf <= sigma_bound   (v2 is short)
        """
        A, B, u = pk
        tau, v = signature

        # Separation of v in the 2 parts
        v1 = v[:self.m1]
        v2 = v[self.m1:]

        # A_tau : Verification matrix [A | tau*G - B]
        A_tau_right = (tau * self.G - B) % self.q
        A_tau = np.hstack([A, A_tau_right])

        # Main equation: A_tau * v == u + D*m mod q
        left_side = np.dot(A_tau.astype(np.int64), v.astype(np.int64)) % self.q
        right_side = (u + np.dot(self.D, message)) % self.q

        equation_matches = np.array_equal(
            left_side.astype(np.int64) % self.q,
            right_side.astype(np.int64) % self.q)

        # Verify that v1 and v2 are smaller than the bound
        v1_is_small = np.max(np.abs(v1)) <= self.sigma1_bound
        v2_is_small = np.max(np.abs(v2)) <= self.sigma_bound

        return equation_matches and v1_is_small and v2_is_small


if __name__ == "__main__":
    print("=" * 60)
    print("  Lattice Signature Scheme")
    print("  SampleD : Micciancio-Peikert (EUROCRYPT 2012)")
    print("=" * 60)

    # 1. Initialisation du systeme
    print("\n[1] Setup...")
    crypto_system = LatticeSignatureScheme()
    print(f"    n={crypto_system.n}, q={crypto_system.q}, "
          f"m1={crypto_system.m1}, m2={crypto_system.m2}, m3={crypto_system.m3}")

    # 2. L'autorite genere ses cles
    print("[2] Keygen...")
    public_key, secret_key = crypto_system.keygen()
    print(f"    Public key: A({public_key[0].shape}), B({public_key[1].shape}), u({public_key[2].shape})")
    print(f"    Secret key: R({secret_key.shape})")

    # 3. Le message a signer (ex: age d'un utilisateur en binaire)
    my_message = np.random.randint(0, 2, size=(crypto_system.m3, 1))
    print(f"[3] Message: vecteur binaire de taille {my_message.shape}")

    # 4. L'autorite signe le message
    tag_id = 42
    print(f"[4] Signature en cours (tag={tag_id})...")
    signature = crypto_system.sign(secret_key, public_key, my_message, tag_id)
    print("    Signature terminee.")

    # 5. Verification de la signature
    print("[5] Verification...")
    is_valid = crypto_system.verify(public_key, my_message, signature)
    print(f"    Verification de la signature : {is_valid}")

    # 6. Infos sur la signature
    tau, v = signature
    v1 = v[:crypto_system.m1]
    v2 = v[crypto_system.m1:]
    print(f"\n--- Statistiques de la signature ---")
    print(f"    ||v1||_inf = {np.max(np.abs(v1)):.0f}  (borne: {crypto_system.sigma1_bound})")
    print(f"    ||v2||_inf = {np.max(np.abs(v2)):.0f}  (borne: {crypto_system.sigma_bound})")
    print(f"    |v| total  = {v.shape[0]} composantes")

    if is_valid:
        print("\n    >>> Signature VALIDE <<<")
    else:
        print("\n    >>> Signature INVALIDE <<<")

    # 7. Test de non-forgeabilite
    print("\n[6] Test de non-forgeabilite...")
    fake_message = np.random.randint(0, 2, size=(crypto_system.m3, 1))
    is_valid_fake = crypto_system.verify(public_key, fake_message, signature)
    print(f"    Verification avec faux message : {is_valid_fake}")
    if not is_valid_fake:
        print("    >>> Faux message correctement rejete <<<")

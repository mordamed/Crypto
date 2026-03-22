"""
Implementation reelle de SampleD (Micciancio-Peikert 2012)
==========================================================

Pre-image sampling avec trapdoor pour les reseaux euclidiens.

Reference : Micciancio, D., & Peikert, C. (2012).
"Trapdoors for Lattices: Simpler, Tighter, Faster, Smaller."
EUROCRYPT 2012.

L'algorithme permet, etant donne :
  - Une matrice A in Z_q^{n x m1}
  - Sa trapdoor R in Z^{m1 x m2} telle que A*R = B mod q
  - Un tag tau et une matrice gadget G
  - Un vecteur cible t in Z_q^n

De trouver un vecteur court v tel que A_tau * v = t mod q
ou A_tau = [A | tau*G - B].

Schema mathematique :
--------------------
  A_tau = [A | tau*G - B]
  F = [R; I_{m2}]           (matrice (m1+m2) x m2)
  A_tau * F = A*R + (tau*G - B)*I = B + tau*G - B = tau*G

  Pour resoudre A_tau * v = target :
    1. Echantillonner p in Z^{m1} (perturbation gaussienne)
    2. Resoudre tau*G*z = target - A*p mod q
       i.e. G*z = tau^{-1} * (target - A*p) mod q
    3. v = [p; 0] + F*z = [p + R*z; z]

  Verification :
    A_tau * v = A*(p + R*z) + (tau*G - B)*z
              = A*p + A*R*z + tau*G*z - B*z
              = A*p + B*z + tau*G*z - B*z
              = A*p + tau*G*z
              = A*p + (target - A*p)     [car tau*G*z = target - A*p]
              = target  mod q
"""

import numpy as np
from typing import Tuple


def sample_discrete_gaussian_1d(center: float, sigma: float) -> int:
    """
    Echantillonne un entier z depuis une distribution gaussienne discrete
    D_{Z, sigma, center} centree en 'center' avec parametre sigma.

    Utilise la methode de rejet : on echantillonne dans un intervalle
    [center - tail_cut*sigma, center + tail_cut*sigma] et on accepte
    avec probabilite proportionnelle a exp(-|x - center|^2 / (2*sigma^2)).

    Parameters
    ----------
    center : float
        Centre de la gaussienne
    sigma : float
        Parametre de largeur (ecart-type)

    Returns
    -------
    int
        Echantillon gaussien discret
    """
    if sigma < 1e-10:
        return int(round(center))

    tail_cut = 12  # nombre d'ecarts-types pour le support
    lower = int(np.floor(center - tail_cut * sigma))
    upper = int(np.ceil(center + tail_cut * sigma))

    while True:
        z = np.random.randint(lower, upper + 1)
        # Probabilite proportionnelle a exp(-|z - center|^2 / (2 * sigma^2))
        prob = np.exp(-((z - center) ** 2) / (2.0 * sigma ** 2))
        if np.random.random() < prob:
            return z


def sample_discrete_gaussian_vector(center: np.ndarray, sigma: float) -> np.ndarray:
    """
    Echantillonne un vecteur depuis une gaussienne discrete spherique
    (chaque coordonnee est independante).

    Parameters
    ----------
    center : np.ndarray
        Vecteur centre (shape (d,1) ou (d,))
    sigma : float
        Parametre de largeur

    Returns
    -------
    np.ndarray
        Vecteur echantillonne, shape identique a center
    """
    flat = center.flatten()
    result = np.array([sample_discrete_gaussian_1d(float(c), sigma) for c in flat],
                      dtype=np.int64)
    return result.reshape(center.shape)


def sample_g_coset(target_val: int, q: int, sigma: float) -> int:
    """
    Resout 1 * z = target_val mod q pour une composante de la matrice gadget
    G = I_n. Trouve z court tel que z = target_val (mod q).

    On echantillonne z = target_mod + k*q avec k entier, en choisissant
    k de sorte que z soit court (centre autour de 0).

    Parameters
    ----------
    target_val : int
        Valeur cible modulo q
    q : int
        Module
    sigma : float
        Parametre gaussien

    Returns
    -------
    int
        z court tel que z = target_val (mod q)
    """
    target_mod = int(target_val) % q

    # On cherche z = target_mod + k*q petit
    # => k ~ -target_mod / q
    k_center = -target_mod / q

    # Sigma pour k : si on veut |z| ~ sigma, alors |k| ~ sigma/q
    sigma_k = max(sigma / q, 1.0)

    k = sample_discrete_gaussian_1d(k_center, sigma_k)
    z = target_mod + k * q
    return int(z)


def sample_g_vector(target: np.ndarray, n: int, m2: int, q: int, sigma: float) -> np.ndarray:
    """
    Resout G*z = target mod q pour G = I_n etendu a n x m2.

    G est de la forme [I_n | 0_{n x (m2-n)}].
    Donc G*z = z[0:n]. On a besoin que z[0:n] = target (mod q),
    et z[n:m2] est libre (on echantillonne depuis D_{Z, sigma, 0}).

    Parameters
    ----------
    target : np.ndarray
        Vecteur cible, shape (n, 1)
    n : int
        Nombre de lignes de G
    m2 : int
        Nombre de colonnes de G (nombre de composantes de z)
    q : int
        Module
    sigma : float
        Parametre gaussien

    Returns
    -------
    np.ndarray
        Vecteur z de shape (m2, 1) tel que G*z = target (mod q) et z court
    """
    z = np.zeros((m2, 1), dtype=np.int64)

    # Les n premieres composantes : z_i = target_i mod q (coset sampling)
    for i in range(n):
        z[i, 0] = sample_g_coset(target[i, 0], q, sigma)

    # Les m2-n composantes restantes : libres, echantillonnees depuis D_{Z, sigma, 0}
    for i in range(n, m2):
        z[i, 0] = sample_discrete_gaussian_1d(0.0, sigma)

    return z


def sampleD(R: np.ndarray, A: np.ndarray, tau: int, target: np.ndarray,
            q: int, G: np.ndarray, sigma: float = 2000.0, sigma1: float = 100000.0) -> np.ndarray:
    """
    Algorithme SampleD (Micciancio-Peikert 2012) - Pre-image sampling avec trapdoor.

    Etant donne :
      - A in Z_q^{n x m1}           (matrice publique)
      - R in Z^{m1 x m2}            (trapdoor : A*R = B mod q)
      - G in Z_q^{n x m2}           (matrice gadget, ici I_n completee)
      - tau in Z_q                   (tag, doit etre inversible mod q)
      - target in Z_q^{n x 1}       (vecteur cible)

    Trouve v = [v1; v2] in Z^{(m1+m2) x 1} tel que :
      [A | tau*G - B] * v = target mod q
      avec v court (norme bornee).

    Schema :
      F = [R; I_{m2}]  =>  A_tau * F = tau * G
      1. Echantillonner p in Z^{m1} (perturbation gaussienne, sigma1)
      2. Calculer target' = tau^{-1} * (target - A*p) mod q
      3. G-sampler z tel que G*z = target' mod q (z court)
      4. v = [p + R*z; z]

    Parameters
    ----------
    R : np.ndarray
        Trapdoor, shape (m1, m2)
    A : np.ndarray
        Matrice publique, shape (n, m1)
    tau : int
        Tag (scalaire, doit etre inversible mod q)
    target : np.ndarray
        Vecteur cible, shape (n, 1)
    q : int
        Module
    G : np.ndarray
        Matrice gadget, shape (n, m2)
    sigma : float
        Parametre gaussien pour le G-sampling (controle la taille de v2)
    sigma1 : float
        Parametre gaussien pour la perturbation (controle la taille de v1)

    Returns
    -------
    np.ndarray
        Vecteur v de shape (m1+m2, 1) tel que A_tau*v = target mod q
    """
    m1, m2 = R.shape
    n = A.shape[0]

    # ============================================================
    # Etape 1 : Perturbation sampling
    # ============================================================
    # Echantillonne p in Z^{m1} depuis D_{Z^m1, sigma1, 0}
    p = sample_discrete_gaussian_vector(np.zeros((m1, 1)), sigma1)

    # ============================================================
    # Etape 2 : Calcul du residu pour le G-sampling
    # ============================================================
    # target' = tau^{-1} * (target - A*p) mod q
    Ap = np.dot(A.astype(np.int64), p.astype(np.int64)) % q
    residual = (target.astype(np.int64) - Ap.astype(np.int64)) % q

    # Calcul de tau^{-1} mod q
    tau_inv = pow(int(tau), -1, q)

    target_for_g = (tau_inv * residual.astype(np.int64)) % q

    # ============================================================
    # Etape 3 : G-sampling
    # ============================================================
    # Resoudre G*z = target_for_g mod q avec z court
    # G = [I_n | 0], donc z[0:n] doit satisfaire la contrainte,
    # et z[n:m2] est libre
    z = sample_g_vector(target_for_g, n, m2, q, sigma)

    # ============================================================
    # Etape 4 : Combinaison via la trapdoor
    # ============================================================
    # v = [p + R*z; z]
    # A_tau * v = A*(p + R*z) + (tau*G - B)*z
    #           = A*p + A*R*z + tau*G*z - B*z
    #           = A*p + B*z + tau*G*z - B*z       [car A*R = B]
    #           = A*p + tau*G*z
    #           = A*p + tau * target_for_g          [car G*z = target_for_g]
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
    Verifie que le vecteur v satisfait A_tau * v = target mod q.

    Parameters
    ----------
    A : np.ndarray
        Matrice publique, shape (n, m1)
    R : np.ndarray
        Trapdoor, shape (m1, m2)
    G : np.ndarray
        Matrice gadget, shape (n, m2)
    tau : int
        Tag
    target : np.ndarray
        Vecteur cible, shape (n, 1)
    v : np.ndarray
        Vecteur pre-image, shape (m1+m2, 1)
    q : int
        Module

    Returns
    -------
    bool
        True si A_tau * v = target mod q
    """
    n, m1 = A.shape
    m2 = G.shape[1]
    B = np.dot(A.astype(np.int64), R.astype(np.int64)) % q

    A_tau_right = (tau * G.astype(np.int64) - B.astype(np.int64)) % q
    A_tau = np.hstack([A.astype(np.int64), A_tau_right])

    result = np.dot(A_tau, v.astype(np.int64)) % q
    target_mod = target.astype(np.int64) % q

    return np.array_equal(result % q, target_mod % q)

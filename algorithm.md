# Micciancio-Peikert (2012) Lattice-Based Signature Scheme

This documentation provides a technical overview of the implementation of the **SampleD** algorithm and its application in a **Hash-and-Sign** signature scheme.

---

## 1. Core Component: The `SampleD` Algorithm

The `SampleD` algorithm is a trapdoor-based preimage sampler. Given a target vector $t$, it finds a short vector $v$ such that $A_\tau \cdot v = t \pmod q$.

### Mathematical Foundations

The system uses an extended public matrix $A_\tau$ defined by a **Tag** ($\tau$) and a **Gadget Matrix** ($G$):
$$A_{\tau} = [A \mid \tau G - B]$$

The algorithm relies on a **Trapdoor Matrix** $R$ such that $A \cdot R = B \pmod q$. To facilitate the math, we define a basis matrix $F$:
$$F = \begin{bmatrix} R \\ I_{m2} \end{bmatrix}$$



### Execution Steps

1.  **Perturbation**: Sample a random Gaussian vector $p \in \mathbb{Z}^{m1}$. This ensures the output distribution does not leak the secret trapdoor $R$.
2.  **Syndrome Calculation**: Compute the residue to be solved by the gadget matrix:
    $$t' = \tau^{-1}(t - A \cdot p) \pmod q$$
3.  **G-Sampling**: Solve $G \cdot z = t' \pmod q$ for a short $z$. Since $G = [I_n \mid 0]$, this simply involves setting $z_i = t'_i$ for the first $n$ components.
4.  **Reconstruction**: Combine the perturbation and the gadget solution:
    $$v = \begin{bmatrix} p + R \cdot z \\ z \end{bmatrix}$$

---

## 2. Application: Lattice Signature Scheme

The `LatticeSignatureScheme` class uses `SampleD` to prove the authenticity of a message $m$.

### System Components

* **$D$ (Commitment Matrix)**: A public matrix used to "bind" the message to the lattice.
* **$u$ (Public Shift)**: A random vector that shifts the target to ensure security.
* **$\tau$ (Tag)**: A unique identifier for the signature context.

### The Protocol Workflow

#### A. Key Generation (`keygen`)
* **Secret Key (SK)**: The trapdoor matrix $R$.
* **Public Key (PK)**: The matrices $A, B$ and the vector $u$.

#### B. Signing (`sign`)
To sign a message $m$:
1.  **Randomize**: Pick a small random vector $r$ (noise).
2.  **Commit**: Calculate $c = A \cdot r + D \cdot m \pmod q$.
3.  **Solve**: Use `sampleD` to find $v'$ such that $A_\tau \cdot v' = u + c \pmod q$.
4.  **Finalize**: The signature is $v = v' - \begin{bmatrix} r \\ 0 \end{bmatrix}$.



#### C. Verification (`verify`)
The verifier checks the signature $v$ against the message $m$ using the public key:

1.  **Algebraic Correctness**:
    The verifier checks if $A_\tau \cdot v \stackrel{?}{=} u + D \cdot m \pmod q$.
    * **Why it works**: Expanding $v$ cancels the random noise $A \cdot r$, leaving exactly $u + D \cdot m$.
2.  **Geometric Correctness**:
    The verifier ensures that the norms $\|v_1\|_\infty$ and $\|v_2\|_\infty$ are within the Gaussian bounds. If the vector is too long, the signature is rejected as a potential forgery.

---

## 3. Security Intuition

* **Hardness**: Without knowing $R$, finding a short vector $v$ that satisfies the verification equation is equivalent to solving the **Shortest Vector Problem (SVP)**, which is computationally infeasible for classical and quantum computers at high dimensions.
* **Unforgeability**: Since $D \cdot m$ changes the target for every different message, a signature for one message cannot be used to forge a signature for another.

---
*Reference: Micciancio, D., & Peikert, C. (2012). "Trapdoors for Lattices: Simpler, Tighter, Faster, Smaller." EUROCRYPT 2012.*
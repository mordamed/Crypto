from math import log2, sqrt, ceil

# reproduces Appendix H (Table H.4) from the paper.

class ParameterAnalyzer:
    @staticmethod
    def compute_paper_params() -> dict:
        """Anonymous credentials."""
        n, d, q = 128, 12, 2**44 - 119
        k, w, kappa = 4, 6, ceil(log2(q))
        m1, m2, ms, m3_attr = 657, d * kappa, 2 * d, 10
        sigma, sigma1, sigma2 = 5609, 26587, 25989

        # Tag space = C(n, w)
        tag_space = 1
        for i in range(w):
            tag_space = tag_space * (n - i) // (i + 1)

        # Size estimates (bits)
        pk_bits = n * ceil(log2(q)) * (d * m1 + d * m2 + d)
        sk_bits = n * ceil(log2(3)) * m1 * m2

        bound_v1 = sigma1 * log2(m1)
        bound_v2 = sigma * log2(m2)
        bits_v1 = ceil(log2(2 * bound_v1 + 1))
        bits_v2 = ceil(log2(2 * bound_v2 + 1))
        sig_bits = n * bits_v1 * m1 + n * bits_v2 * m2 + w * ceil(log2(n))

        # SIS norm bound beta_I (Lemma 3.5)
        beta_I_sq = (1 + (sqrt(n*m1) + sqrt(n*m2) + 7.5)**2) * (sigma1**2 * n * m1 + sigma**2 * n * m2)
        beta_I_sq += ((sqrt(n*m1) + sqrt(n*m3_attr) + 7.5) * sqrt(n * m3_attr) + 1)**2

        return {
            'n': n, 'd': d, 'q': q, 'k': k, 'w': w,
            'm1': m1, 'm2': m2, 'ms': ms, 'm3': m3_attr,
            'sigma': sigma, 'sigma1': sigma1, 'sigma2': sigma2,
            'tag_space': tag_space, 'log2_tag_space': log2(tag_space),
            'pk_MB': pk_bits / (8 * 1024**2),
            'sk_MB': sk_bits / (8 * 1024**2),
            'sig_KB': sig_bits / (8 * 1024),
            'proof_KB': 724,                    # From Table H.4
            'beta_I': sqrt(beta_I_sq),
            'bound_v1': bound_v1, 'bound_v2': bound_v2,
        }

    @staticmethod
    def print_table_1_1():
        """Print Table 1.1 comparison with [LLM+16]."""
        print("\n" + "=" * 78)
        print("  TABLE 1.1 — Comparison with [LLM+16]")
        print("=" * 78)

        rows = [
            ("|pk|",         "707·10⁴ MB", "296·10 MB",  "7.8 MB"),
            ("|sk|",         "372·10² MB", "229·10 MB",  "8.9 MB"),
            ("|sig|",        "2139·10² KB",   "418 KB",   "273 KB"),
            ("|π| (ZK proof)", "2671·10³ KB", "177·10² KB", "639 KB"),
        ]

        print(f"\n  {'Metric':<20} {'LLM+16 (fast)':>18} {'Paper (standard)':>18} {'Paper (module)':>18}")
        print("  " + "-" * 74)
        for row in rows:
            print(f"  {row[0]:<20} {row[1]:>18} {row[2]:>18} {row[3]:>18}")

        print("\n  Improvement factors (module vs LLM+16):")
        print(f"    Public key:  ~900x   |  Secret key: ~4000x")
        print(f"    Signature:   ~780x   |  ZK proof:   ~4000x")

    @staticmethod
    def print_evoting_analysis():
        """Print e-voting scalability analysis."""
        params = ParameterAnalyzer.compute_paper_params()

        print("\n" + "=" * 78)
        print("  PARAMETER ANALYSIS FOR E-VOTING (Table H.4)")
        print("=" * 78)

        print(f"\n  Lattice: n={params['n']}, d={params['d']}, "
              f"q≈2^{log2(params['q']):.0f}, k={params['k']}, w={params['w']}")
        print(f"  Dims:    m1={params['m1']}, m2={params['m2']}, "
              f"ms={params['ms']}, m3={params['m3']}")
        print(f"  Widths:  σ={params['sigma']}, σ1={params['sigma1']}, σ2={params['sigma2']}")
        print(f"\n  |pk|={params['pk_MB']:.2f} MB, |sk|={params['sk_MB']:.2f} MB, "
              f"|sig|≈{params['sig_KB']:.0f} KB, |π|={params['proof_KB']} KB")

        print(f"\n  E-Voting Scalability (proof size per voter = {params['proof_KB']} KB):")
        for nv in [1_000, 10_000, 100_000, 1_000_000]:
            total = nv * params['proof_KB'] / 1024
            print(f"    {nv:>10,} voters → {total:,.0f} MB ({total/1024:.1f} GB)")

        print(f"\n  vs Helios (pairing-based): ~{params['proof_KB'] // 3}x overhead, "
              f"BUT Helios is not post-quantum")
        print(f"  Tag space C(128,6) ≈ 2^{params['log2_tag_space']:.1f} "
              f"→ {params['tag_space']:,} voters max (any national election)")

        return params

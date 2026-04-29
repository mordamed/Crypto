import sys
import numpy as np
import time
from .evoting import ElectionAuthority, BulletinBoard, Voter
from .parameters import ParameterAnalyzer

# Force UTF-8 output on Windows so special characters render correctly
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

"""
Main simulation for the lattice-based e-voting scheme. The simulation is divided into phases:
    1. Setup: The election authority initializes the system parameters and generates the public key.
    2. Registration: Voters register with the authority and receive their credentials (signatures).
    3. Voting: Registered voters cast their votes, which are verified and recorded on the bulletin board
    4. Double-vote attack: An attempt to vote twice is made, which should be detected and rejected.
    5. Forgery attack: An attempt to forge a signature is made, which should be rejected based on the hardness of the underlying lattice problem.
    6. Tally: The votes are tallied, and the results are then displayed.
"""


def run_simulation():
    print("=" * 78)
    print("  LATTICE-BASED E-VOTING SIMULATION")
    print("=" * 78)

    candidates = ["Alice Wonderland", "Bob Builder", "Charlie Chaplin"]
    n_voters = 99999

    # -- Phase 1: Setup --
    print("\n-- PHASE 1: ELECTION SETUP --")
    authority = ElectionAuthority(candidates, n_voters)
    board = BulletinBoard()
    scheme = authority.scheme

    print(f"  Lattice params: n={scheme.n}, q={scheme.q}, m1={scheme.m1}, m2={scheme.m2}")
    print(f"  |pk| = {(authority.pk['A'].nbytes + authority.pk['B'].nbytes + authority.pk['u'].nbytes) / 1024:.1f} KB")

    # -- Phase 2: Registration --
    print("\n-- PHASE 2: VOTER REGISTRATION --")
    voters = []
    for vid in range(n_voters):
        district = np.random.randint(0, 2, size=(4, 1), dtype=np.int64)
        eligible = np.ones((4, 1), dtype=np.int64)
        attrs = np.vstack([district, eligible])

        t0 = time.time()
        cred = authority.issue_credential(vid, attrs)
        dt = (time.time() - t0) * 1000

        if cred:
            voters.append(Voter(vid, cred, scheme))
            if vid < 3 or vid == n_voters - 1:
                print(f"  Voter {vid:2d}: tag={cred.tag:3d}, "
                      f"|sig|={cred.signature['v'].nbytes / 1024:.1f} KB, {dt:.0f} ms")
            elif vid == 3:
                print(f"  ...")

    print(f"  Total registered: {len(voters)}")

    # -- Phase 3: Voting --
    print("\n-- PHASE 3: VOTE CASTING --")
    np.random.seed(42)
    for voter in voters:
        chosen = np.random.randint(0, len(candidates))
        t0 = time.time()
        ballot = voter.cast_vote(chosen, authority.pk)
        dt = (time.time() - t0) * 1000

        accepted = board.submit_ballot(ballot) if ballot else False
        status = "OK" if accepted else "REJECTED"

        if voter.voter_id < 3 or voter.voter_id == n_voters - 1:
            print(f"  Voter {voter.voter_id:2d} -> {candidates[chosen]:18s} "
                  f"| proof={'OK' if ballot.proof_valid else 'FAIL':4s} | {status} ({dt:.0f} ms)")
        elif voter.voter_id == 3:
            print(f"  ...")

    # -- Phase 4: Double-vote attack --
    print("\n-- PHASE 4: DOUBLE-VOTE DETECTION --")
    attackerTarget = 41
    attacker = voters[attackerTarget]
    ballot2 = attacker.cast_vote(3, authority.pk)
    accepted = board.submit_ballot(ballot2) if ballot2 else False
    print(f"  Voter {attackerTarget} re-votes with tag={ballot2.tag}: "
          f"{'ACCEPTED' if accepted else 'REJECTED (duplicate tag detected)'}")

    # -- Phase 5: Forgery attack --
    print("\n-- PHASE 5: FORGERY ATTEMPT --")
    fake_m = np.random.randint(0, 2, size=(scheme.m3, 1), dtype=np.int64)
    fake_v = np.random.randint(-100, 100, size=(scheme.m, 1), dtype=np.int64)
    forged = scheme.verify(authority.pk, fake_m, {'tau': 999, 'v': fake_v})
    print(f"  Forged signature: {'ACCEPTED' if forged else 'REJECTED (SIS hardness)'}")

    # -- Phase 6: Tally --
    print("\n-- PHASE 6: RESULTS --")
    result = board.tally(candidates)
    winner, best = "", 0
    max_count = max(result.vote_counts.values()) if result.vote_counts else 1
    for c, count in result.vote_counts.items():
        bar = "#" * round(count / max_count * 40)
        print(f"  {c:<22} {count:>6}  {bar}")
        if count > best:
            best, winner = count, c
    print(f"\n  Valid: {result.total_valid} | Rejected: {result.total_rejected} "
          f"| Double-votes: {result.double_votes_detected}")
    print(f"  WINNER: {winner} ({best} votes)")

    print(f"""
-- SECURITY PROPERTIES --
  1. ANONYMITY     - bulletin board shows only tag + vote, not identity
  2. UNFORGEABILITY - forgery {'rejected' if not forged else 'FAILED'} (EUF-CMA -> SIS)
  3. DOUBLE-VOTE   - {result.double_votes_detected} attempt(s) detected via tag uniqueness
  4. POST-QUANTUM  - based on Module-SIS/LWE (128-bit quantum security)
""")
    return result


if __name__ == "__main__":
    run_simulation()
    ParameterAnalyzer.print_table_1_1()
    ParameterAnalyzer.print_evoting_analysis()

    print("\n" + "=" * 78)
    print("  DONE")
    print("=" * 78)

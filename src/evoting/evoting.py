"""
E-Voting System based on Lattice Anonymous Credentials

Maps Section 5.3 of the paper to an electronic voting scenario:
    Organization  →  ElectionAuthority
    OKeyGen       →  Election setup
    UKeyGen       →  Voter registration (voter generates secret key)
    Issue         →  Credential issuance (oblivious signing, Algorithm 5.5)
    Show          →  Vote casting (ZK proof of credential, Algorithm 5.6)
"""

import numpy as np
import time
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Dict

from .signature import LatticeSignatureScheme


@dataclass
class VoterCredential:
    """A voter's anonymous credential = SEP signature on (secret_key || attributes)"""
    voter_id: int
    secret_key: np.ndarray
    attributes: np.ndarray
    signature: dict
    tag: int


@dataclass
class CastBallot:
    """A ballot posted to the public bulletin board"""
    ballot_id: str
    vote: int
    tag: int
    proof_valid: bool
    timestamp: float


@dataclass
class ElectionResult:
    """Final election tally"""
    candidates: List[str]
    vote_counts: Dict[str, int]
    total_valid: int
    total_rejected: int
    double_votes_detected: int


class ElectionAuthority:
    """
    Election Authority — the Organization in Section 5.3.

    Generates election keys (OKeyGen = Algorithm 5.3) and issues
    voter credentials (Issue = Algorithm 5.5). Cannot link
    credentials to votes after issuance (anonymity property).
    """

    def __init__(self, candidates: List[str], n_voters: int):
        self.candidates = candidates
        self.n_voters = n_voters
        self.ms = 8
        self.m_attr = 8
        m3 = self.ms + self.m_attr

        self.scheme = LatticeSignatureScheme(n=64, q=3329, m1=128, m3=m3)
        self.pk, self.sk = self.scheme.keygen()

        # Stateful tag counter: tau = F(st), F injective (Section 3.1)
        self.state = 1
        self.registration_table: Dict[int, int] = {}

    def issue_credential(self, voter_id: int, attributes: np.ndarray) -> Optional[VoterCredential]:
        """
        Algorithm 5.5: Issue — Credential Issuance Protocol

        In the full protocol (Algorithm 5.1: OblSign):
        1. User commits to (s || m) with randomness r'
        2. User proves knowledge of (r', s) via ZK
        3. Authority adds randomness r'' and signs
        4. User absorbs r' to get final signature

        The tag tau is unique per voter — prevents double-voting
        while preserving anonymity.
        """
        if voter_id in self.registration_table:
            return None

        # Algorithm 5.4: UKeyGen
        s = np.random.randint(0, 2, size=(self.ms, 1), dtype=np.int64)
        m_e = np.vstack([s, attributes])

        tag = self.state
        self.state += 1

        sig = self.scheme.sign(self.sk, self.pk, m_e, tag)
        self.registration_table[voter_id] = tag

        return VoterCredential(
            voter_id=voter_id,
            secret_key=s,
            attributes=attributes,
            signature=sig,
            tag=tag
        )


class BulletinBoard:
    """
    Public bulletin board storing cast ballots.

    Enforces tag uniqueness (double-vote prevention) and
    proof validity (public verifiability).
    """

    def __init__(self):
        self.ballots: List[CastBallot] = []
        self.used_tags: set = set()
        self.rejected_count = 0
        self.double_vote_count = 0

    def submit_ballot(self, ballot: CastBallot) -> bool:
        if ballot.tag in self.used_tags:
            self.double_vote_count += 1
            self.rejected_count += 1
            return False
        if not ballot.proof_valid:
            self.rejected_count += 1
            return False

        self.ballots.append(ballot)
        self.used_tags.add(ballot.tag)
        return True

    def tally(self, candidates: List[str]) -> ElectionResult:
        counts = {c: 0 for c in candidates}
        for ballot in self.ballots:
            if 0 <= ballot.vote < len(candidates):
                counts[candidates[ballot.vote]] += 1
        return ElectionResult(
            candidates=candidates,
            vote_counts=counts,
            total_valid=len(self.ballots),
            total_rejected=self.rejected_count,
            double_votes_detected=self.double_vote_count
        )


class Voter:
    """
    A voter holding an anonymous credential.
    Uses the Show protocol (Algorithm 5.6) to cast a vote.
    """

    def __init__(self, voter_id: int, credential: VoterCredential, scheme: LatticeSignatureScheme):
        self.voter_id = voter_id
        self.credential = credential
        self.scheme = scheme
        self.has_voted = False

    def cast_vote(self, vote: int, pk: dict) -> Optional[CastBallot]:
        """
        Algorithm 5.6: Show — Credential Showing Protocol

        The real protocol proves in ZK that:
            Verify(pk, m_e, (tau, v), pp) = 1
        revealing ONLY the tag (for double-vote check) and the vote.
        Identity, secret key, other attributes remain hidden.

        Here we simulate by running the actual verify.
        """
        m_e = np.vstack([
            self.credential.secret_key,
            self.credential.attributes,
        ])
        proof_valid = self.scheme.verify(pk, m_e, self.credential.signature)

        ballot_hash = hashlib.sha256(
            f"{self.credential.tag}:{vote}:{time.time()}".encode()
        ).hexdigest()[:16]

        self.has_voted = True
        return CastBallot(
            ballot_id=ballot_hash,
            vote=vote,
            tag=self.credential.tag,
            proof_valid=proof_valid,
            timestamp=time.time()
        )

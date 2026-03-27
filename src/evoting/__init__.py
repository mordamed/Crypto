"""
Lattice Signature with Efficient Protocols — E-Voting Application

Paper: Jeudy, Roux-Langlois, Sanders (Crypto 2023)
Implementation ref: truncated-sampler (Jeudy & Sanders, Asiacrypt 2025)
"""

from .lattice_sampler import sampleD, discrete_gaussian_sample, gadget_solve
from .signature import LatticeSignatureScheme
from .evoting import ElectionAuthority, BulletinBoard, Voter, VoterCredential, CastBallot, ElectionResult
from .parameters import ParameterAnalyzer

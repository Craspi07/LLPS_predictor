"""
π-interaction features (PScore-like).

PScore quantifies the density of π-π and cation-π interactions along a
sequence.  High PScore correlates strongly with phase separation driven by
multivalent aromatic interactions.

References:
  Vernon et al. (2018) eLife — PScore definition
  Liang et al. (2024) EMBO J — MolPhase
"""

import numpy as np

# Aromatic residues that form π-π stacking
AROMATIC = frozenset("YFWH")

# Cation residues involved in cation-π interactions
CATION = frozenset("RKH")

# All π-active residues (union — H appears in both)
PI_ACTIVE = frozenset("YFWHRK")


def _pair_counts(sequence: str, group_a: frozenset, group_b: frozenset, max_dist: int = 10) -> int:
    """Count residue pairs of group_a × group_b within sequence distance max_dist."""
    n = len(sequence)
    count = 0
    for i, aa_i in enumerate(sequence):
        if aa_i in group_a:
            lo = max(0, i - max_dist)
            hi = min(n, i + max_dist + 1)
            for j in range(lo, hi):
                if j != i and sequence[j] in group_b:
                    count += 1
    return count


def compute_pscore(sequence: str, max_dist: int = 10) -> float:
    """
    Sequence-based PScore approximation.

    PScore ≈ (2 × aromatic_pairs + cation_pi_pairs) / length

    Aromatic–aromatic pairs are weighted ×2 because they contribute two
    π-electrons each; cation–π pairs are weighted ×1.
    """
    n = len(sequence)
    aromatic_pairs = _pair_counts(sequence, AROMATIC, AROMATIC, max_dist)
    cation_pi_pairs = _pair_counts(sequence, CATION, AROMATIC, max_dist)
    # avoid double-counting aromatic–aromatic symmetric pairs
    aromatic_pairs //= 2
    return (2 * aromatic_pairs + cation_pi_pairs) / n


def compute_pi_features(sequence: str, max_dist: int = 10) -> dict:
    """Return all π-interaction features."""
    n = len(sequence)

    # Densities
    aromatic_density = sum(1 for aa in sequence if aa in AROMATIC) / n
    cation_density = sum(1 for aa in sequence if aa in CATION) / n
    pi_active_density = sum(1 for aa in sequence if aa in PI_ACTIVE) / n

    # Pair counts
    aromatic_pairs = _pair_counts(sequence, AROMATIC, AROMATIC, max_dist) // 2
    cation_pi_pairs = _pair_counts(sequence, CATION, AROMATIC, max_dist)
    # Tyrosine specifically — strongest π-stacker in IDRs
    y_density = sequence.count("Y") / n
    r_density = sequence.count("R") / n  # Arginine cation-π is the dominant driver

    pscore = (2 * aromatic_pairs + cation_pi_pairs) / n

    return {
        "pscore_approx": pscore,
        "aromatic_density": aromatic_density,
        "cation_density": cation_density,
        "pi_active_density": pi_active_density,
        "aromatic_aromatic_pairs": aromatic_pairs,
        "cation_pi_pairs": cation_pi_pairs,
        "tyrosine_density": y_density,
        "arginine_density": r_density,
        "yr_ratio": y_density / max(r_density, 1e-6),  # Y/R balance
    }

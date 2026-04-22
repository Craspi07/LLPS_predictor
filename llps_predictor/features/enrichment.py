"""
Amino-acid enrichment features (PSAP-like, prion-like domain signature).

Prion-like low-complexity domains (PLDs) rich in G, S, Q, N, Y are a major
driver of LLPS in RNA-binding proteins.

References:
  Lancaster et al. (2014) Cell — prion-like domain importance
  Alberti et al. (2009) Cell — prion-like enrichment
  PSAP: Saar et al. (2021) — phase-separation-associated protein classifier
"""

import numpy as np
from collections import Counter
from math import log2

# Residues enriched in prion-like / low-complexity domains
PLD_RESIDUES = frozenset("GSQNY")

# Residues associated specifically with RNA-granule-forming proteins
GRANULE_RESIDUES = frozenset("GSQNYFR")

# Residues with backbone/sidechain amide groups (H-bond donors in β-pairing)
BETA_PRONE = frozenset("IVLYFWT")

# Standard amino-acid background frequencies (Swiss-Prot 2023)
BACKGROUND_FREQ = {
    "A": 0.0825, "R": 0.0553, "N": 0.0406, "D": 0.0545, "C": 0.0137,
    "Q": 0.0393, "E": 0.0675, "G": 0.0707, "H": 0.0227, "I": 0.0596,
    "L": 0.0966, "K": 0.0584, "M": 0.0242, "F": 0.0386, "P": 0.0470,
    "S": 0.0656, "T": 0.0534, "W": 0.0108, "Y": 0.0292, "V": 0.0687,
}


def _enrichment_ratio(observed: float, background: float) -> float:
    """Log2 enrichment ratio with pseudo-count protection."""
    return log2((observed + 1e-6) / (background + 1e-6))


def compute_sequence_complexity(sequence: str) -> float:
    """
    Linguistic complexity / Shannon entropy normalised to [0,1].
    Low complexity (close to 0) is a hallmark of disordered phase-separating IDRs.
    """
    n = len(sequence)
    counts = Counter(sequence)
    entropy = -sum((c / n) * log2(c / n) for c in counts.values() if c > 0)
    max_entropy = log2(min(n, 20))  # 20 amino acids
    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_enrichment_features(sequence: str) -> dict:
    """
    Return amino-acid enrichment features relevant to LLPS.

    Includes:
      - PSAP-like score (G/S/Q/N/Y content)
      - Prion-like domain (PLD) fraction
      - Q/N content (glutamine/asparagine — polar zippers)
      - Low-complexity score (inverse of Shannon entropy)
      - Per-residue enrichment ratios for key LLPS-driver residues
    """
    n = len(sequence)
    counts = Counter(sequence)
    freq = {aa: counts.get(aa, 0) / n for aa in "ACDEFGHIKLMNPQRSTVWY"}

    psap_enrichment = sum(freq.get(aa, 0) for aa in PLD_RESIDUES)
    granule_enrichment = sum(freq.get(aa, 0) for aa in GRANULE_RESIDUES)
    qn_content = freq.get("Q", 0) + freq.get("N", 0)
    gs_content = freq.get("G", 0) + freq.get("S", 0)
    aromatic_content = sum(freq.get(aa, 0) for aa in "YFWH")
    charged_content = sum(freq.get(aa, 0) for aa in "RKDE")

    # Enrichment ratios relative to Swiss-Prot background
    enrichment_Y = _enrichment_ratio(freq.get("Y", 0), BACKGROUND_FREQ["Y"])
    enrichment_R = _enrichment_ratio(freq.get("R", 0), BACKGROUND_FREQ["R"])
    enrichment_Q = _enrichment_ratio(freq.get("Q", 0), BACKGROUND_FREQ["Q"])
    enrichment_N = _enrichment_ratio(freq.get("N", 0), BACKGROUND_FREQ["N"])
    enrichment_G = _enrichment_ratio(freq.get("G", 0), BACKGROUND_FREQ["G"])
    enrichment_S = _enrichment_ratio(freq.get("S", 0), BACKGROUND_FREQ["S"])

    lc_score = 1.0 - compute_sequence_complexity(sequence)  # high = low complexity

    # Unique residue fraction (simple diversity measure)
    unique_fraction = len(counts) / 20.0

    return {
        "psap_enrichment": psap_enrichment,
        "granule_enrichment": granule_enrichment,
        "qn_content": qn_content,
        "gs_content": gs_content,
        "aromatic_content": aromatic_content,
        "charged_content": charged_content,
        "low_complexity_score": lc_score,
        "sequence_complexity": 1.0 - lc_score,
        "unique_residue_fraction": unique_fraction,
        "enrichment_Y": enrichment_Y,
        "enrichment_R": enrichment_R,
        "enrichment_Q": enrichment_Q,
        "enrichment_N": enrichment_N,
        "enrichment_G": enrichment_G,
        "enrichment_S": enrichment_S,
        "residue_frequencies": freq,
    }

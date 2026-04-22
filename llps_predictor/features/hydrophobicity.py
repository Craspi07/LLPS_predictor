"""
Hydrophobicity features using the Kyte-Doolittle scale (1982).

References:
  Kyte & Doolittle (1982) J Mol Biol
  Wilson et al. (2023) ParSe 2.0 — sliding window approach
"""

import numpy as np
from typing import List

# Kyte-Doolittle hydrophobicity scale
KD_SCALE = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}

# Eisenberg consensus hydrophobicity (alternative normalised scale)
EISENBERG_SCALE = {
    "A":  0.620, "R": -2.530, "N": -0.780, "D": -0.900, "C":  0.290,
    "Q": -0.850, "E": -0.740, "G":  0.480, "H": -0.400, "I":  1.380,
    "L":  1.060, "K": -1.500, "M":  0.640, "F":  1.190, "P":  0.120,
    "S": -0.180, "T": -0.050, "W":  0.810, "Y":  0.260, "V":  1.080,
}


def _score_array(sequence: str, scale: dict) -> np.ndarray:
    return np.array([scale.get(aa, 0.0) for aa in sequence])


def sliding_window_hydrophobicity(
    sequence: str,
    window_size: int = 25,
    scale: dict = None,
) -> List[float]:
    """Return mean hydrophobicity for each window position (Kyte-Doolittle by default)."""
    if scale is None:
        scale = KD_SCALE
    scores = _score_array(sequence, scale)
    n = len(scores)
    if n < window_size:
        return [float(np.mean(scores))]
    return [float(np.mean(scores[i : i + window_size])) for i in range(n - window_size + 1)]


def compute_hydrophobicity_features(sequence: str, window_size: int = 25) -> dict:
    """
    Compute hydrophobicity features used as LLPS predictors.

    Low mean hydrophobicity combined with high disorder is a hallmark of
    IDR-driven phase separation.  High local hydrophobicity peaks can indicate
    hydrophobic sticker residues that drive droplet formation.
    """
    scores = _score_array(sequence, KD_SCALE)
    profile = sliding_window_hydrophobicity(sequence, window_size=window_size)

    mean_h = float(np.mean(scores))
    max_local_h = float(np.max(profile))
    min_local_h = float(np.min(profile))
    std_h = float(np.std(scores))

    # Fraction of hydrophobic residues (KD > 1.5)
    hydrophobic_residues = {"A", "C", "F", "I", "L", "M", "V", "W"}
    f_hydrophobic = sum(1 for aa in sequence if aa in hydrophobic_residues) / len(sequence)

    # Amphipathicity proxy: variance in the sliding-window profile
    profile_variance = float(np.var(profile)) if len(profile) > 1 else 0.0

    # Normalised mean hydrophobicity: map to [0,1] using KD range [-4.5, 4.5]
    norm_mean_h = (mean_h + 4.5) / 9.0

    return {
        "mean_hydrophobicity": mean_h,
        "max_local_hydrophobicity": max_local_h,
        "min_local_hydrophobicity": min_local_h,
        "hydrophobicity_std": std_h,
        "fraction_hydrophobic": f_hydrophobic,
        "window_profile_variance": profile_variance,
        "mean_hydrophobicity_normalised": norm_mean_h,
        "hydrophobicity_profile": profile,
        "window_size": window_size,
    }

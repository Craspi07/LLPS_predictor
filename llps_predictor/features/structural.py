"""
Secondary-structure propensity features (Chou-Fasman scales).

β-pairing propensity (Mullick & Trovato 2022) is an independent predictor
of LLPS; proteins that form transient β-zippers show enhanced droplet stability.

References:
  Chou & Fasman (1978) Ann Rev Biochem
  Mullick & Trovato (2022) Biomolecules
"""

import numpy as np
from typing import List

# Chou-Fasman secondary structure propensity scales
# P_alpha: α-helix propensity
CF_ALPHA = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
    "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}

# P_beta: β-sheet propensity
CF_BETA = {
    "A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19,
    "Q": 1.10, "E": 0.37, "G": 0.75, "H": 0.87, "I": 1.60,
    "L": 1.30, "K": 0.74, "M": 1.05, "F": 1.38, "P": 0.55,
    "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.70,
}

# P_turn: β-turn propensity
CF_TURN = {
    "A": 0.66, "R": 0.95, "N": 1.56, "D": 1.46, "C": 1.19,
    "Q": 0.98, "E": 0.74, "G": 1.56, "H": 0.95, "I": 0.47,
    "L": 0.59, "K": 1.01, "M": 0.60, "F": 0.60, "P": 1.52,
    "S": 1.43, "T": 0.96, "W": 0.96, "Y": 1.14, "V": 0.50,
}

# β-pairing propensity scale (Mullick & Trovato 2022, Table S1)
# Captures tendency to form inter-strand hydrogen bonds (β-zippers)
BETA_PAIRING = {
    "A": 0.50, "R": 0.80, "N": 0.42, "D": 0.26, "C": 0.74,
    "Q": 0.96, "E": 0.26, "G": 0.40, "H": 0.82, "I": 1.20,
    "L": 1.10, "K": 0.65, "M": 0.90, "F": 1.38, "P": 0.10,
    "S": 0.59, "T": 0.90, "W": 1.47, "Y": 1.45, "V": 1.55,
}


def _window_mean(sequence: str, scale: dict, window: int) -> List[float]:
    scores = np.array([scale.get(aa, 1.0) for aa in sequence])
    n = len(scores)
    if n < window:
        return [float(np.mean(scores))]
    return [float(np.mean(scores[i : i + window])) for i in range(n - window + 1)]


def compute_structural_features(sequence: str, window: int = 9) -> dict:
    """
    Compute secondary-structure propensity features.

    Key output:
      mean_beta_propensity — overall β-sheet tendency (Chou-Fasman)
      mean_beta_pairing    — β-pairing/zipper propensity (Mullick & Trovato)
      mean_alpha_propensity
      helix_vs_sheet_ratio — α/β balance
    """
    alpha_scores = np.array([CF_ALPHA.get(aa, 1.0) for aa in sequence])
    beta_scores = np.array([CF_BETA.get(aa, 1.0) for aa in sequence])
    turn_scores = np.array([CF_TURN.get(aa, 1.0) for aa in sequence])
    bp_scores = np.array([BETA_PAIRING.get(aa, 0.5) for aa in sequence])

    mean_alpha = float(np.mean(alpha_scores))
    mean_beta = float(np.mean(beta_scores))
    mean_turn = float(np.mean(turn_scores))
    mean_bp = float(np.mean(bp_scores))

    alpha_profile = _window_mean(sequence, CF_ALPHA, window)
    beta_profile = _window_mean(sequence, CF_BETA, window)
    bp_profile = _window_mean(sequence, BETA_PAIRING, window)

    # Fraction of windows predicted as helix/sheet (propensity > 1.0 = above average)
    frac_helix_windows = float(np.mean([s > 1.0 for s in alpha_profile]))
    frac_beta_windows = float(np.mean([s > 1.0 for s in beta_profile]))

    helix_vs_sheet = mean_alpha / mean_beta if mean_beta > 0 else 0.0

    return {
        "mean_alpha_propensity": mean_alpha,
        "mean_beta_propensity": mean_beta,
        "mean_turn_propensity": mean_turn,
        "mean_beta_pairing": mean_bp,
        "max_local_beta_pairing": float(np.max(bp_profile)),
        "frac_helix_windows": frac_helix_windows,
        "frac_beta_windows": frac_beta_windows,
        "helix_vs_sheet_ratio": helix_vs_sheet,
        "beta_pairing_profile": bp_profile,
        "alpha_profile": alpha_profile,
        "beta_profile": beta_profile,
    }

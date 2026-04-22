"""
Charge-patterning features: SCD, kappa, NCPR, FCR.

References:
  Das & Pappu (2013) PNAS — kappa definition
  Sawle & Ghosh (2015) J Chem Phys — SCD definition
"""

import numpy as np

CHARGES = {"R": 1, "K": 1, "D": -1, "E": -1}


def _charge_array(sequence: str) -> np.ndarray:
    return np.array([CHARGES.get(aa, 0) for aa in sequence], dtype=float)


def compute_scd(sequence: str) -> float:
    """
    Sequence Charge Decoration (SCD).
    SCD = (1/N) Σ_{i<j} q_i * q_j * sqrt(j - i)
    More negative SCD → charges of opposite sign are closer → stronger electrostatic attraction.
    """
    q = _charge_array(sequence)
    n = len(q)
    scd = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            scd += q[i] * q[j] * np.sqrt(j - i)
    return scd / n


def compute_kappa(sequence: str, window_size: int = 5) -> float:
    """
    Charge patterning parameter κ (Das & Pappu 2013).
    κ ∈ [0, 1]; 0 = well-mixed charges, 1 = fully segregated.
    High κ drives chain collapse; moderate κ promotes LLPS.

    Uses the sliding-window approximation of the original definition.
    """
    q = _charge_array(sequence)
    n = len(q)
    if n == 0:
        return 0.0

    f_pos = np.sum(q > 0) / n
    f_neg = np.sum(q < 0) / n
    fcr = f_pos + f_neg
    if fcr == 0:
        return 0.0

    # δ: mean squared deviation of local charge fraction from global NCPR
    ncpr = (f_pos - f_neg)
    deltas = []
    half = window_size // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window = q[lo:hi]
        local_ncpr = np.sum(window) / len(window)
        deltas.append((local_ncpr - ncpr) ** 2)
    delta = np.mean(deltas)

    # δ_max for a fully charge-segregated sequence
    # When all + come first then all –:  δ_max ≈ fcr²/4
    delta_max = (fcr ** 2) / 4.0 if fcr > 0 else 1.0
    kappa = delta / delta_max if delta_max > 0 else 0.0
    return float(min(kappa, 1.0))


def compute_ncpr(sequence: str) -> float:
    """Net Charge Per Residue = (f+ − f−) / length."""
    q = _charge_array(sequence)
    return float(np.sum(q) / len(q))


def compute_fcr(sequence: str) -> float:
    """Fraction of Charged Residues = (n+ + n−) / length."""
    q = _charge_array(sequence)
    return float(np.sum(np.abs(q)) / len(q))


def compute_charge_features(sequence: str) -> dict:
    """Return all charge-patterning features as a dict."""
    q = _charge_array(sequence)
    n = len(sequence)
    return {
        "scd": compute_scd(sequence),
        "kappa": compute_kappa(sequence),
        "ncpr": float(np.sum(q) / n),
        "fcr": float(np.sum(np.abs(q)) / n),
        "n_positive": int(np.sum(q > 0)),
        "n_negative": int(np.sum(q < 0)),
        "charge_asymmetry": float(abs(np.sum(q > 0) - np.sum(q < 0)) / max(1, np.sum(np.abs(q)))),
    }

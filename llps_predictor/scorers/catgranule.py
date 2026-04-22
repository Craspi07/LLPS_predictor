"""
catGRANULE score — classic formula and 2.0-inspired approximation.

catGRANULE (Agostini et al. 2013) predicts granule-forming propensity using:
  G = (w_RNA × R_RNA + w_dis × R_dis + w_pi × R_pi) × log10(L) - offset

catGRANULE 2.0 (Bose et al. 2022 / Zenodo 14205831) extends this with
AlphaFold2 structural features via a 128-feature MLP.  The full model
requires the ROBOT command-line package; this module implements the
classic formula and a structural-feature-enriched approximation.

References:
  Agostini et al. (2013) Nucleic Acids Res 41:e201
  Bose et al. (2022) Mol Cell 82:4293
"""

import math
import numpy as np
from typing import Optional

# RNA-binding propensity scale (Agostini et al. 2013, Supp. Table S1)
# Positive = enriched in granule-forming / RNA-binding proteins
RNA_BINDING_PROP = {
    "R":  0.0943, "K":  0.0607, "G":  0.0412, "S":  0.0323, "H":  0.0189,
    "N":  0.0145, "Y":  0.0112, "F":  0.0078, "W":  0.0067, "T":  0.0056,
    "Q":  0.0034, "C":  0.0021, "A":  0.0012, "M":  0.0008, "V": -0.0034,
    "L": -0.0045, "I": -0.0056, "P": -0.0067, "D": -0.0089, "E": -0.0112,
}

# catGRANULE classic model weights (fitted from Agostini et al. 2013)
W_RNA = 0.693    # RNA-binding contribution weight
W_DIS = 0.831    # Disorder contribution weight
W_PI  = 0.256    # π-interaction contribution weight
OFFSET = 1.391   # Decision boundary; G > 0 → granule-forming

PI_RESIDUES = frozenset("YFWRH")  # π-system interactors


def compute_catgranule_classic(
    sequence: str,
    disorder_fraction: Optional[float] = None,
) -> dict:
    """
    Compute the classic catGRANULE (2013) score.

    disorder_fraction: pre-computed IUPred3 value; if None, a heuristic is used.
    Score > 0 → predicted granule-forming protein.
    """
    n = len(sequence)
    if n == 0:
        return {"catgranule_score": 0.0, "method": "classic"}

    # --- RNA binding propensity ---
    r_rna = sum(RNA_BINDING_PROP.get(aa, 0.0) for aa in sequence) / n

    # --- Structural disorder ---
    if disorder_fraction is None:
        disorder_promoting = frozenset("AERDGKPQS")
        disorder_fraction = sum(1 for aa in sequence if aa in disorder_promoting) / n

    r_dis = disorder_fraction

    # --- π-interaction density ---
    r_pi = sum(1 for aa in sequence if aa in PI_RESIDUES) / n

    # catGRANULE formula
    log_len = math.log10(n)
    score = (W_RNA * r_rna + W_DIS * r_dis + W_PI * r_pi) * log_len - OFFSET

    return {
        "catgranule_score": float(score),
        "catgranule_rna_component": float(W_RNA * r_rna * log_len),
        "catgranule_disorder_component": float(W_DIS * r_dis * log_len),
        "catgranule_pi_component": float(W_PI * r_pi * log_len),
        "rna_binding_propensity": float(r_rna),
        "pi_density_catg": float(r_pi),
        "log_length": float(log_len),
        "method": "classic",
    }


def compute_catgranule_v2_approx(
    sequence: str,
    disorder_fraction: Optional[float] = None,
    mean_beta_propensity: Optional[float] = None,
    mean_alpha_propensity: Optional[float] = None,
) -> dict:
    """
    catGRANULE 2.0-inspired approximation.

    Extends the classic formula with:
      - β-sheet propensity (from Chou-Fasman; structural context)
      - α-helix propensity (ordered regions reduce LLPS propensity)
      - Prion-like enrichment (G/S/Q/N/Y composition)

    The structural parameters are optional; when absent, the function
    gracefully degrades to the classic formula.
    """
    classic = compute_catgranule_classic(sequence, disorder_fraction)

    # Prion-like composition bonus (Q/N/G/S/Y enrichment)
    pld_residues = frozenset("GSQNY")
    pld_fraction = sum(1 for aa in sequence if aa in pld_residues) / len(sequence)

    # Structural modifiers from AlphaFold-derived features (if available)
    struct_modifier = 0.0
    if mean_beta_propensity is not None:
        # β-sheet propensity above 1.0 (average) slightly increases score
        struct_modifier += 0.15 * (mean_beta_propensity - 1.0)
    if mean_alpha_propensity is not None:
        # α-helix propensity above 1.0 slightly decreases score (more ordered)
        struct_modifier -= 0.10 * (mean_alpha_propensity - 1.0)

    score_v2 = classic["catgranule_score"] + 0.25 * pld_fraction + struct_modifier

    result = dict(classic)
    result.update({
        "catgranule_score": float(score_v2),
        "catgranule_score_classic": classic["catgranule_score"],
        "pld_fraction": float(pld_fraction),
        "structural_modifier": float(struct_modifier),
        "method": "v2_approx",
    })
    return result


def get_catgranule_score(
    sequence: str,
    disorder_fraction: Optional[float] = None,
    mean_beta_propensity: Optional[float] = None,
    mean_alpha_propensity: Optional[float] = None,
    catgranule_bin: Optional[str] = None,
) -> dict:
    """
    Master entry point for catGRANULE scoring.

    If a local catGRANULE ROBOT binary is provided and accessible, it is
    called via subprocess (the binary must accept --input / --output flags).
    Otherwise, the v2 approximation is returned.
    """
    import os, subprocess, tempfile, csv

    if catgranule_bin and os.path.isfile(catgranule_bin):
        # Write FASTA and call the binary
        fd, fasta = tempfile.mkstemp(suffix=".fasta")
        out = fasta.replace(".fasta", "_cat.csv")
        with os.fdopen(fd, "w") as f:
            f.write(f">protein\n{sequence}\n")
        try:
            proc = subprocess.run(
                [catgranule_bin, "--input", fasta, "--output", out],
                capture_output=True, text=True, timeout=120,
            )
            if proc.returncode == 0:
                with open(out, newline="") as cf:
                    reader = csv.DictReader(cf)
                    for row in reader:
                        for key in ("Score", "score", "catGRANULE", "catgranule"):
                            if key in row:
                                return {
                                    "catgranule_score": float(row[key]),
                                    "method": "robot_binary",
                                }
        except Exception:
            pass
        finally:
            for p in (fasta, out):
                try:
                    os.remove(p)
                except OSError:
                    pass

    return compute_catgranule_v2_approx(
        sequence,
        disorder_fraction=disorder_fraction,
        mean_beta_propensity=mean_beta_propensity,
        mean_alpha_propensity=mean_alpha_propensity,
    )

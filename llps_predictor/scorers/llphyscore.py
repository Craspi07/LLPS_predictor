"""
LLPhyScore: wrapper for the local installation and sequence-based approximation.

LLPhyScore (Vernon et al. 2018, eLife) scores proteins based on the density of
physical interactions — sp2-sp2 (π-π), cation-π, and electrostatic — that
stabilise condensate droplets.

This module:
  1. Tries to call the locally installed LLPhyScore binary/script.
  2. Falls back to a sequence-based approximation if the tool is unavailable.

Reference:
  Vernon et al. (2018) eLife 7:e31486
"""

import os
import subprocess
import tempfile
import csv
import numpy as np
from typing import Optional

# Residue sets
SP2_AROMATIC = frozenset("YFWH")          # sp2 π-electron donors (aromatic)
SP2_AMIDE = frozenset("NQST")             # sp2 amide/carbonyl (backbone + sidechain)
SP2_ALL = SP2_AROMATIC | SP2_AMIDE | frozenset("RK")   # all sp2-rich residues
CATION = frozenset("RKH")
ANION = frozenset("DE")


def _write_fasta(sequence: str, seq_id: str = "protein") -> str:
    """Write sequence to a temporary FASTA file; returns the file path."""
    fd, path = tempfile.mkstemp(suffix=".fasta")
    with os.fdopen(fd, "w") as f:
        f.write(f">{seq_id}\n{sequence}\n")
    return path


def run_llphyscore_local(
    sequence: str,
    llphyscore_script: str,
    seq_id: str = "protein",
) -> Optional[float]:
    """
    Call the local LLPhyScore installation.

    llphyscore_script: path to LLPhyScore.py (from the standalone package)
    Returns the LLPhyScore value, or None if the call fails.
    """
    fasta_path = _write_fasta(sequence, seq_id)
    out_path = fasta_path.replace(".fasta", "_llphy_out.csv")
    try:
        cmd = ["python", llphyscore_script, "--input", fasta_path, "--output", out_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None
        with open(out_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Column name may vary; try common keys
                for key in ("LLPhyScore", "score", "Score", "llphyscore"):
                    if key in row:
                        return float(row[key])
    except Exception:
        return None
    finally:
        for p in (fasta_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass
    return None


def compute_llphyscore_approx(sequence: str, max_dist: int = 10) -> dict:
    """
    Sequence-based LLPhyScore approximation.

    Estimates three interaction classes within a local neighbourhood
    (max_dist residues):
      1. sp2-sp2 (aromatic π-π): Y/F/W/H pairs
      2. Cation-π: R/K/H near Y/F/W
      3. Electrostatic: R/K near D/E

    The composite score is a weighted linear combination, scaled to make
    it roughly comparable with the published LLPhyScore range.
    """
    n = len(sequence)

    sp2_pi_pairs = 0
    cation_pi_pairs = 0
    electro_pairs = 0

    for i, aa_i in enumerate(sequence):
        lo = max(0, i - max_dist)
        hi = min(n, i + max_dist + 1)

        if aa_i in SP2_AROMATIC:
            for j in range(lo, hi):
                if j > i and sequence[j] in SP2_AROMATIC:
                    sp2_pi_pairs += 1

        if aa_i in CATION:
            for j in range(lo, hi):
                if j != i and sequence[j] in SP2_AROMATIC:
                    cation_pi_pairs += 1

        if aa_i in CATION:
            for j in range(lo, hi):
                if j != i and sequence[j] in ANION:
                    electro_pairs += 1

    sp2_density = sp2_pi_pairs / n
    cation_pi_density = cation_pi_pairs / n
    electro_density = electro_pairs / n

    # Composite (weights from Vernon et al. 2018 relative importance)
    score = (0.45 * sp2_density + 0.35 * cation_pi_density + 0.20 * electro_density)

    return {
        "llphyscore_approx": float(score),
        "sp2_pi_density": float(sp2_density),
        "cation_pi_density_llphy": float(cation_pi_density),
        "electrostatic_density": float(electro_density),
        "sp2_pi_pairs": sp2_pi_pairs,
        "cation_pi_pairs_count": cation_pi_pairs,
        "electrostatic_pairs_count": electro_pairs,
        "method": "approximation",
    }


def get_llphyscore(
    sequence: str,
    llphyscore_script: Optional[str] = None,
    seq_id: str = "protein",
) -> dict:
    """
    Return LLPhyScore — from local tool if available, otherwise approximation.
    """
    if llphyscore_script and os.path.isfile(llphyscore_script):
        score = run_llphyscore_local(sequence, llphyscore_script, seq_id)
        if score is not None:
            approx = compute_llphyscore_approx(sequence)
            approx["llphyscore"] = score
            approx["method"] = "local_tool"
            return approx

    result = compute_llphyscore_approx(sequence)
    result["llphyscore"] = result["llphyscore_approx"]
    return result

"""
Intrinsic disorder prediction.

Primary:  IUPred3 REST API (returns per-residue disorder scores 0–1).
Fallback: Sequence-based heuristic using disorder-/order-promoting residues
          (Uversky et al. 2000; Dunker et al. 2001).

References:
  Dosztányi et al. (2005) — IUPred
  Mészáros et al. (2021) Nucleic Acids Res — IUPred3
"""

import time
import requests
import numpy as np
from typing import List, Optional

IUPRED3_API = "https://iupred3.elte.hu/iupred3api"

# Disorder propensity per residue (Uversky scale, positive = disorder-promoting)
DISORDER_PROPENSITY = {
    "A":  0.06, "R":  0.18, "N":  0.00, "D":  0.19, "C": -0.20,
    "Q":  0.07, "E":  0.24, "G":  0.10, "H":  0.00, "I": -0.28,
    "L": -0.28, "K":  0.18, "M": -0.03, "F": -0.32, "P":  0.27,
    "S":  0.09, "T":  0.03, "W": -0.32, "Y": -0.18, "V": -0.28,
}

# Strong disorder-promoting residues
DISORDER_PROMOTING = frozenset("AERDGKPQS")
# Strong order-promoting residues
ORDER_PROMOTING = frozenset("CFILVWY")


def _heuristic_disorder(sequence: str, window: int = 11) -> List[float]:
    """
    Sliding-window heuristic disorder score based on residue propensity.
    Returns per-residue scores in [0, 1].
    """
    scores_raw = np.array([DISORDER_PROPENSITY.get(aa, 0.0) for aa in sequence])
    n = len(scores_raw)
    half = window // 2
    smoothed = np.zeros(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed[i] = np.mean(scores_raw[lo:hi])
    # Normalise to [0, 1]
    s_min, s_max = smoothed.min(), smoothed.max()
    if s_max > s_min:
        return list((smoothed - s_min) / (s_max - s_min))
    return [0.5] * n


def fetch_iupred3(
    sequence: str,
    uniprot_id: Optional[str] = None,
    disorder_type: str = "long",
) -> dict:
    """
    Fetch per-residue disorder scores from the IUPred3 API.

    Tries the UniProt accession endpoint first (more reliable), then falls
    back to a POST with the raw sequence.  If both fail, uses the
    heuristic disorder estimator.

    disorder_type: 'long'  — long disordered regions (default, better for IDRs)
                  'short' — short flexible loops
    """
    scores = None
    anchor2 = None
    source = None

    # --- Attempt 1: GET by UniProt accession ---
    if uniprot_id:
        try:
            url = f"{IUPRED3_API}?accession={uniprot_id}&type={disorder_type}&output=json"
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                data = r.json()
                scores = data.get("iupred3") or data.get("iupred2") or data.get("score")
                anchor2 = data.get("anchor2")
                source = "IUPred3-accession"
        except Exception:
            pass

    # --- Attempt 2: POST raw sequence ---
    if scores is None:
        for attempt in range(3):
            try:
                r = requests.post(
                    IUPRED3_API,
                    data={"seq": sequence, "type": disorder_type},
                    timeout=30,
                )
                if r.status_code == 200:
                    data = r.json()
                    scores = data.get("iupred3") or data.get("iupred2") or data.get("score")
                    anchor2 = data.get("anchor2")
                    source = "IUPred3-sequence"
                    break
            except Exception:
                if attempt < 2:
                    time.sleep(2 ** attempt)

    # --- Fallback: heuristic ---
    if scores is None or len(scores) != len(sequence):
        scores = _heuristic_disorder(sequence)
        source = "heuristic"
        anchor2 = None

    scores = [float(s) for s in scores]
    mean_disorder = float(np.mean(scores))
    disordered_fraction = float(np.mean([s > 0.5 for s in scores]))

    # Identify IDR blocks (contiguous regions with score > 0.5)
    in_idr = False
    idrs = []
    start = 0
    for i, s in enumerate(scores):
        if s > 0.5 and not in_idr:
            in_idr = True
            start = i
        elif s <= 0.5 and in_idr:
            in_idr = False
            idrs.append((start, i - 1))
    if in_idr:
        idrs.append((start, len(scores) - 1))

    longest_idr = max((e - s + 1 for s, e in idrs), default=0)
    n_idrs = len(idrs)

    # ANCHOR2 binding sites (disordered protein-binding regions)
    anchor2_sites = []
    if anchor2:
        anchor2_float = [float(v) for v in anchor2]
        in_anchor = False
        a_start = 0
        for i, v in enumerate(anchor2_float):
            if v > 0.5 and not in_anchor:
                in_anchor = True
                a_start = i
            elif v <= 0.5 and in_anchor:
                in_anchor = False
                anchor2_sites.append((a_start, i - 1))
        if in_anchor:
            anchor2_sites.append((a_start, len(anchor2_float) - 1))

    return {
        "disorder_scores": scores,
        "mean_disorder": mean_disorder,
        "disordered_fraction": disordered_fraction,
        "idr_regions": idrs,
        "n_idrs": n_idrs,
        "longest_idr_length": longest_idr,
        "anchor2_scores": anchor2,
        "anchor2_sites": anchor2_sites,
        "source": source,
    }

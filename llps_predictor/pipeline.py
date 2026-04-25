"""
Main LLPS prediction pipeline orchestrator.

Accepts a UniProt accession or gene symbol, fetches the sequence,
runs all feature extractors and scorers, and produces an ensemble score.
"""

import math
from typing import Optional

from .fetcher import fetch_sequence
from .features.charge import compute_charge_features
from .features.hydrophobicity import compute_hydrophobicity_features
from .features.pi_score import compute_pi_features
from .features.enrichment import compute_enrichment_features
from .features.disorder import fetch_iupred3
from .features.structural import compute_structural_features
from .scorers.llphyscore import get_llphyscore
from .scorers.catgranule import get_catgranule_score


# ---------------------------------------------------------------------------
# Sigmoid helper for normalisation
# ---------------------------------------------------------------------------

def _sigmoid(x: float, center: float = 0.0, scale: float = 1.0) -> float:
    return 1.0 / (1.0 + math.exp(-(x - center) / scale))


# ---------------------------------------------------------------------------
# Ensemble score
# ---------------------------------------------------------------------------

def _compute_ensemble(features: dict) -> dict:
    """
    Combine individual scores into an ensemble LLPS propensity score in [0, 1].

    Each component is first normalised independently to [0, 1]:
      disorder_fraction : already 0-1
      catGRANULE        : sigmoid centred at 0 (decision boundary), scale 1.5
      LLPhyScore approx : sigmoid centred at 0.04, scale 0.02
      PScore approx     : sigmoid centred at 0.5, scale 0.3
      PSAP enrichment   : already 0-1 (G/S/Q/N/Y fraction)
      |SCD|             : sigmoid centred at 5, scale 3
      Low complexity    : already 0-1

    The final score is the simple (unweighted) average of all normalised
    component scores — i.e. the mean rank across all metrics.
    """
    components = {
        "disorder":       features["disorder"]["disordered_fraction"],
        "catgranule":     _sigmoid(features["catgranule"]["catgranule_score"], 0, 1.5),
        "llphyscore":     _sigmoid(features["llphyscore"]["llphyscore_approx"], 0.04, 0.02),
        "pscore":         _sigmoid(features["pi"]["pscore_approx"], 0.5, 0.3),
        "psap":           features["enrichment"]["psap_enrichment"],
        "scd":            _sigmoid(abs(features["charge"]["scd"]), 5.0, 3.0),
        "low_complexity": features["enrichment"]["low_complexity_score"],
    }

    ensemble = sum(components.values()) / len(components)

    return {
        "ensemble_score": round(ensemble, 4),
        "component_scores": {k: round(v, 4) for k, v in components.items()},
        "n_components": len(components),
    }


def _classify(ensemble_score: float) -> dict:
    """Map ensemble score to a human-readable LLPS classification."""
    if ensemble_score >= 0.70:
        label = "Very High"
        interpretation = (
            "Strong evidence for LLPS. This protein likely phase separates under "
            "physiological conditions. Multiple biophysical drivers detected."
        )
    elif ensemble_score >= 0.50:
        label = "High"
        interpretation = (
            "Good evidence for LLPS propensity. Phase separation probable, "
            "potentially requiring specific partner proteins or conditions."
        )
    elif ensemble_score >= 0.35:
        label = "Moderate"
        interpretation = (
            "Moderate LLPS propensity. Some features consistent with phase "
            "separation; experimental validation recommended."
        )
    else:
        label = "Low"
        interpretation = (
            "Limited evidence for LLPS. Protein likely does not phase separate "
            "under standard conditions."
        )
    return {"label": label, "interpretation": interpretation}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    identifier: str,
    organism: str = "human",
    llphyscore_script: Optional[str] = None,
    catgranule_bin: Optional[str] = None,
    disorder_type: str = "long",
    hydro_window: int = 25,
    sequence: Optional[str] = None,
) -> dict:
    """
    Run the complete LLPS prediction pipeline.

    Parameters
    ----------
    identifier      : UniProt accession (e.g. 'Q15653') or gene symbol (e.g. 'TDP43')
    organism        : organism name for gene-symbol lookup (default 'human')
    llphyscore_script : path to local LLPhyScore.py (optional)
    catgranule_bin  : path to catGRANULE ROBOT binary (optional)
    disorder_type   : 'long' (default) or 'short' for IUPred3
    hydro_window    : sliding window size for hydrophobicity (default 25)
    sequence        : if provided, skip the UniProt fetch and use this sequence directly
                      (identifier is still used as the protein label)

    Returns a nested dict with all features and the ensemble score.
    """
    # 1. Fetch / accept sequence
    if sequence:
        from .fetcher import validate_sequence
        seq = validate_sequence(sequence)
        protein_info = {
            "accession": identifier,
            "gene_name": identifier,
            "protein_name": identifier,
            "organism": organism,
            "sequence": seq,
            "length": len(seq),
            "reviewed": False,
            "source": "user_provided",
        }
    else:
        protein_info = fetch_sequence(identifier, organism=organism)
        seq = protein_info["sequence"]

    uniprot_id = protein_info.get("accession")

    # 2. Feature extraction (run all)
    charge_feat = compute_charge_features(seq)
    hydro_feat = compute_hydrophobicity_features(seq, window_size=hydro_window)
    pi_feat = compute_pi_features(seq)
    enrich_feat = compute_enrichment_features(seq)
    struct_feat = compute_structural_features(seq)
    disorder_feat = fetch_iupred3(seq, uniprot_id=uniprot_id, disorder_type=disorder_type)

    # 3. Scoring
    llphy_result = get_llphyscore(seq, llphyscore_script=llphyscore_script, seq_id=uniprot_id or identifier)
    catg_result = get_catgranule_score(
        seq,
        disorder_fraction=disorder_feat["disordered_fraction"],
        mean_beta_propensity=struct_feat["mean_beta_propensity"],
        mean_alpha_propensity=struct_feat["mean_alpha_propensity"],
        catgranule_bin=catgranule_bin,
    )

    # 4. Ensemble
    feature_bundle = {
        "charge": charge_feat,
        "hydrophobicity": hydro_feat,
        "pi": pi_feat,
        "enrichment": enrich_feat,
        "structural": struct_feat,
        "disorder": disorder_feat,
        "llphyscore": llphy_result,
        "catgranule": catg_result,
    }
    ensemble = _compute_ensemble(feature_bundle)
    classification = _classify(ensemble["ensemble_score"])

    # 5. Build full result
    result = {
        "protein_info": protein_info,
        "features": feature_bundle,
        "scoring": {
            "llphyscore": llphy_result,
            "catgranule": catg_result,
            "ensemble": ensemble,
            "classification": classification,
        },
        "summary": {
            "identifier": identifier,
            "uniprot_id": uniprot_id,
            "gene_name": protein_info.get("gene_name", ""),
            "protein_name": protein_info.get("protein_name", ""),
            "organism": protein_info.get("organism", ""),
            "length": len(seq),
            "ensemble_score": ensemble["ensemble_score"],
            "llps_classification": classification["label"],
            "interpretation": classification["interpretation"],
            "disorder_source": disorder_feat["source"],
            "key_metrics": {
                "disordered_fraction": round(disorder_feat["disordered_fraction"], 3),
                "n_idrs": disorder_feat["n_idrs"],
                "longest_idr": disorder_feat["longest_idr_length"],
                "mean_disorder": round(disorder_feat["mean_disorder"], 3),
                "scd": round(charge_feat["scd"], 3),
                "kappa": round(charge_feat["kappa"], 3),
                "fcr": round(charge_feat["fcr"], 3),
                "ncpr": round(charge_feat["ncpr"], 3),
                "pscore_approx": round(pi_feat["pscore_approx"], 4),
                "aromatic_density": round(pi_feat["aromatic_density"], 3),
                "psap_enrichment": round(enrich_feat["psap_enrichment"], 3),
                "qn_content": round(enrich_feat["qn_content"], 3),
                "low_complexity": round(enrich_feat["low_complexity_score"], 3),
                "catgranule_score": round(catg_result["catgranule_score"], 3),
                "llphyscore_approx": round(llphy_result["llphyscore_approx"], 4),
                "mean_hydrophobicity": round(hydro_feat["mean_hydrophobicity"], 3),
                "mean_beta_pairing": round(struct_feat["mean_beta_pairing"], 3),
            },
        },
    }
    return result

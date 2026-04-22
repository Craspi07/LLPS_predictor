"""
Report generation — terminal (rich) and file outputs (JSON, TSV, CSV).
"""

import json
import csv
import os
from typing import Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich import box
    from rich.text import Text
    from rich.rule import Rule
    _RICH = True
except ImportError:
    _RICH = False


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _score_colour(score: float) -> str:
    if score >= 0.70:
        return "bold red"
    elif score >= 0.50:
        return "bold yellow"
    elif score >= 0.35:
        return "yellow"
    return "green"


def _label_colour(label: str) -> str:
    colours = {"Very High": "bold red", "High": "bold yellow", "Moderate": "yellow", "Low": "green"}
    return colours.get(label, "white")


# ---------------------------------------------------------------------------
# Rich terminal report
# ---------------------------------------------------------------------------

def print_report(result: dict, console: Optional[object] = None) -> None:
    """Print a rich, formatted report to the terminal."""
    if not _RICH:
        _print_plain(result)
        return

    console = console or Console()
    summary = result["summary"]
    scoring = result["scoring"]
    ensemble = scoring["ensemble"]["ensemble_score"]
    label = summary["llps_classification"]
    km = summary["key_metrics"]

    # ── Header ────────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]LLPS Propensity Prediction Report[/bold cyan]")
    console.print()

    # Protein info panel
    info_lines = [
        f"[bold]UniProt ID  :[/bold] {summary.get('uniprot_id', 'N/A')}",
        f"[bold]Gene Name   :[/bold] {summary.get('gene_name', 'N/A')}",
        f"[bold]Protein     :[/bold] {summary.get('protein_name', 'N/A')}",
        f"[bold]Organism    :[/bold] {summary.get('organism', 'N/A')}",
        f"[bold]Length      :[/bold] {summary['length']} aa",
        f"[bold]Disorder src:[/bold] {summary.get('disorder_source', 'N/A')}",
    ]
    console.print(Panel("\n".join(info_lines), title="Protein Information", border_style="cyan"))

    # ── Ensemble Score Banner ─────────────────────────────────────────────
    colour = _score_colour(ensemble)
    label_colour = _label_colour(label)
    score_text = Text()
    score_text.append(f"  Ensemble LLPS Score: ", style="bold")
    score_text.append(f"{ensemble:.4f}", style=colour)
    score_text.append(f"   │   Classification: ", style="bold")
    score_text.append(f"{label}", style=label_colour)
    score_text.append(f"  ", style="bold")
    console.print(Panel(score_text, border_style=colour))

    console.print(f"  [italic]{summary['interpretation']}[/italic]")
    console.print()

    # ── Ensemble Component Table ──────────────────────────────────────────
    comp_table = Table(title="Ensemble Score Components", box=box.SIMPLE_HEAVY, show_header=True)
    comp_table.add_column("Component", style="cyan", width=20)
    comp_table.add_column("Normalised Score", justify="right", width=18)
    comp_table.add_column("Weight", justify="right", width=10)
    comp_table.add_column("Contribution", justify="right", width=14)

    components = scoring["ensemble"]["component_scores"]
    weights = scoring["ensemble"]["component_weights"]
    for comp, score in sorted(components.items(), key=lambda x: -x[1]):
        w = weights[comp]
        contrib = score * w
        colour_c = _score_colour(score)
        comp_table.add_row(
            comp.replace("_", " ").title(),
            f"[{colour_c}]{score:.4f}[/{colour_c}]",
            f"{w:.2f}",
            f"{contrib:.4f}",
        )
    console.print(comp_table)

    # ── Key Biophysical Metrics ───────────────────────────────────────────
    met_table = Table(title="Key Biophysical Metrics", box=box.SIMPLE, show_header=True)
    met_table.add_column("Metric", style="cyan", width=30)
    met_table.add_column("Value", justify="right", width=12)
    met_table.add_column("Metric", style="cyan", width=30)
    met_table.add_column("Value", justify="right", width=12)

    rows = [
        ("Disordered Fraction", f"{km['disordered_fraction']:.3f}"),
        ("Mean Disorder (IUPred3)", f"{km['mean_disorder']:.3f}"),
        ("# IDR Regions", str(km["n_idrs"])),
        ("Longest IDR (aa)", str(km["longest_idr"])),
        ("SCD", f"{km['scd']:.3f}"),
        ("κ (kappa)", f"{km['kappa']:.3f}"),
        ("FCR", f"{km['fcr']:.3f}"),
        ("NCPR", f"{km['ncpr']:.3f}"),
        ("PScore (approx)", f"{km['pscore_approx']:.4f}"),
        ("Aromatic Density", f"{km['aromatic_density']:.3f}"),
        ("PSAP Enrichment (G/S/Q/N/Y)", f"{km['psap_enrichment']:.3f}"),
        ("Q+N Content", f"{km['qn_content']:.3f}"),
        ("Low Complexity Score", f"{km['low_complexity']:.3f}"),
        ("Mean Hydrophobicity (KD)", f"{km['mean_hydrophobicity']:.3f}"),
        ("catGRANULE Score", f"{km['catgranule_score']:.3f}"),
        ("LLPhyScore (approx)", f"{km['llphyscore_approx']:.4f}"),
        ("Mean β-Pairing Propensity", f"{km['mean_beta_pairing']:.3f}"),
    ]

    # Print in two columns
    for i in range(0, len(rows), 2):
        r1 = rows[i]
        r2 = rows[i + 1] if i + 1 < len(rows) else ("", "")
        met_table.add_row(r1[0], r1[1], r2[0], r2[1])
    console.print(met_table)

    # ── Individual Scorer Results ─────────────────────────────────────────
    catg = scoring["catgranule"]
    llphy = scoring["llphyscore"]
    console.print(Panel(
        f"[bold]Score:[/bold] {catg['catgranule_score']:.3f}  "
        f"(> 0 → granule-forming)\n"
        f"  RNA component:      {catg.get('catgranule_rna_component', 'N/A')}\n"
        f"  Disorder component: {catg.get('catgranule_disorder_component', 'N/A')}\n"
        f"  π component:        {catg.get('catgranule_pi_component', 'N/A')}\n"
        f"  Method:             {catg.get('method', 'N/A')}",
        title="catGRANULE Score",
        border_style="blue",
    ))
    console.print(Panel(
        f"[bold]Score (approx):[/bold] {llphy['llphyscore_approx']:.4f}\n"
        f"  sp2 π-π density:       {llphy.get('sp2_pi_density', 'N/A')}\n"
        f"  Cation-π density:      {llphy.get('cation_pi_density_llphy', 'N/A')}\n"
        f"  Electrostatic density: {llphy.get('electrostatic_density', 'N/A')}\n"
        f"  Method:                {llphy.get('method', 'N/A')}",
        title="LLPhyScore",
        border_style="magenta",
    ))

    # ── IDR Regions ───────────────────────────────────────────────────────
    idrs = result["features"]["disorder"].get("idr_regions", [])
    if idrs:
        idr_table = Table(title="Intrinsically Disordered Regions (IUPred3 > 0.5)", box=box.SIMPLE)
        idr_table.add_column("Region #", justify="right")
        idr_table.add_column("Start (1-indexed)", justify="right")
        idr_table.add_column("End (1-indexed)", justify="right")
        idr_table.add_column("Length (aa)", justify="right")
        for idx, (s, e) in enumerate(idrs, 1):
            idr_table.add_row(str(idx), str(s + 1), str(e + 1), str(e - s + 1))
        console.print(idr_table)
    else:
        console.print("[dim]No IDR regions detected (or all scores ≤ 0.5).[/dim]")

    console.print()
    console.rule("[dim]End of Report[/dim]")
    console.print()


def _print_plain(result: dict) -> None:
    """Plain-text fallback when rich is unavailable."""
    s = result["summary"]
    print("=" * 60)
    print("LLPS Propensity Prediction")
    print("=" * 60)
    print(f"UniProt ID : {s.get('uniprot_id', 'N/A')}")
    print(f"Gene       : {s.get('gene_name', 'N/A')}")
    print(f"Protein    : {s.get('protein_name', 'N/A')}")
    print(f"Organism   : {s.get('organism', 'N/A')}")
    print(f"Length     : {s['length']} aa")
    print(f"Ensemble   : {s['ensemble_score']}")
    print(f"Class      : {s['llps_classification']}")
    print(f"Note       : {s['interpretation']}")
    print("-" * 60)
    for k, v in s["key_metrics"].items():
        print(f"  {k:<30} {v}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# File export
# ---------------------------------------------------------------------------

def save_json(result: dict, path: str) -> None:
    """Save full result dict as JSON (profiles/arrays excluded for readability)."""
    clean = _strip_profiles(result)
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"JSON report saved → {path}")


def save_tsv(result: dict, path: str) -> None:
    """Save a flat TSV of the key metrics + ensemble score."""
    s = result["summary"]
    row = {
        "uniprot_id": s.get("uniprot_id", ""),
        "gene_name": s.get("gene_name", ""),
        "protein_name": s.get("protein_name", ""),
        "organism": s.get("organism", ""),
        "length": s["length"],
        "ensemble_score": s["ensemble_score"],
        "llps_classification": s["llps_classification"],
    }
    row.update(s["key_metrics"])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    print(f"TSV report saved → {path}")


def save_disorder_profile(result: dict, path: str) -> None:
    """Save per-residue disorder + β-pairing scores as TSV."""
    seq = result["protein_info"]["sequence"]
    dis = result["features"]["disorder"]["disorder_scores"]
    bp = result["features"]["structural"]["beta_pairing_profile"]
    alpha_p = result["features"]["structural"]["alpha_profile"]
    # Extend profiles to full length (window profiles are shorter)
    n = len(seq)

    def _pad(profile, length):
        if len(profile) == length:
            return profile
        # Repeat last value to fill
        extended = list(profile)
        while len(extended) < length:
            extended.append(extended[-1])
        return extended[:length]

    bp_full = _pad(bp, n)
    alpha_full = _pad(alpha_p, n)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["position", "residue", "disorder_score", "beta_pairing", "alpha_propensity"])
        for i, aa in enumerate(seq):
            writer.writerow([
                i + 1, aa,
                round(dis[i], 4) if i < len(dis) else "",
                round(bp_full[i], 4),
                round(alpha_full[i], 4),
            ])
    print(f"Per-residue profile saved → {path}")


def _strip_profiles(obj):
    """Recursively remove large list fields (profiles) to keep JSON readable."""
    _skip_keys = {
        "disorder_scores", "anchor2_scores", "hydrophobicity_profile",
        "beta_pairing_profile", "alpha_profile", "beta_profile",
        "residue_frequencies",
    }
    if isinstance(obj, dict):
        return {k: _strip_profiles(v) for k, v in obj.items() if k not in _skip_keys}
    if isinstance(obj, list) and len(obj) > 50:
        return f"[array of {len(obj)} values]"
    return obj

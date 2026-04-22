#!/usr/bin/env python3
"""
LLPS Predictor — Command-Line Interface

Predict liquid-liquid phase separation (LLPS) propensity from a UniProt
accession ID or gene symbol.

Examples
--------
# By gene symbol (human, default)
python main.py TDP43

# By UniProt accession
python main.py Q15653

# Specify organism
python main.py FUS --organism human

# Save outputs
python main.py hnRNPA1 --json results/hnRNPA1.json --tsv results/hnRNPA1.tsv --profile results/hnRNPA1_profile.tsv

# With local tool paths
python main.py FUS --llphyscore /path/to/LLPhyScore.py --catgranule /path/to/catgranule_robot

# Provide sequence directly (skip UniProt fetch)
python main.py MY_PROTEIN --seq MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQ

# Batch mode (text file with one identifier per line)
python main.py --batch proteins.txt --tsv batch_results.tsv
"""

import argparse
import sys
import os
import json

# Make sure the package is importable when running from the repo root
sys.path.insert(0, os.path.dirname(__file__))

from llps_predictor.pipeline import run_pipeline
from llps_predictor.report import print_report, save_json, save_tsv, save_disorder_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="llps_predictor",
        description="Predict LLPS propensity from a UniProt ID or gene symbol.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "identifier",
        nargs="?",
        help="UniProt accession (e.g. Q15653) or gene symbol (e.g. TDP43).",
    )
    input_group.add_argument(
        "--batch",
        metavar="FILE",
        help="Text file with one identifier per line for batch prediction.",
    )

    parser.add_argument(
        "--seq",
        metavar="SEQUENCE",
        help="Provide amino acid sequence directly (one-letter codes, no header). "
             "Skips the UniProt fetch; 'identifier' is used as the protein label.",
    )
    parser.add_argument(
        "--fasta",
        metavar="FILE",
        help="Read sequence(s) from a FASTA file. "
             "If the file contains multiple entries they are all analysed (batch mode).",
    )
    parser.add_argument(
        "--organism",
        default="human",
        metavar="ORGANISM",
        help="Organism for gene-symbol lookup (default: human). "
             "Examples: mouse, yeast, ecoli.",
    )

    # External tools
    parser.add_argument(
        "--llphyscore",
        metavar="PATH",
        help="Path to the local LLPhyScore.py script (optional).",
    )
    parser.add_argument(
        "--catgranule",
        metavar="PATH",
        help="Path to the local catGRANULE ROBOT binary (optional).",
    )

    # Algorithm options
    parser.add_argument(
        "--disorder-type",
        choices=["long", "short"],
        default="long",
        help="IUPred3 disorder type (default: long).",
    )
    parser.add_argument(
        "--hydro-window",
        type=int,
        default=25,
        metavar="N",
        help="Sliding window size for hydrophobicity (default: 25).",
    )

    # Output
    parser.add_argument(
        "--json",
        metavar="FILE",
        help="Save full results as JSON.",
    )
    parser.add_argument(
        "--tsv",
        metavar="FILE",
        help="Save key metrics as TSV (append in batch mode).",
    )
    parser.add_argument(
        "--profile",
        metavar="FILE",
        help="Save per-residue disorder + structural profile as TSV.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Suppress terminal report (useful for batch scripting).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all output except errors and file-save confirmations.",
    )

    return parser.parse_args()


def run_single(
    identifier: str,
    args: argparse.Namespace,
    tsv_append: bool = False,
) -> dict:
    """Run pipeline for a single identifier and handle output."""
    if not args.quiet:
        print(f"\nFetching & analysing: {identifier!r} ...")

    result = run_pipeline(
        identifier=identifier,
        organism=args.organism,
        llphyscore_script=args.llphyscore,
        catgranule_bin=args.catgranule,
        disorder_type=args.disorder_type,
        hydro_window=args.hydro_window,
        sequence=getattr(args, "seq", None),
    )

    if not args.no_report and not args.quiet:
        print_report(result)

    if args.json:
        save_json(result, args.json)

    if args.tsv:
        # In batch mode, append rows after the first
        _save_tsv_batch(result, args.tsv, append=tsv_append)

    if args.profile:
        save_disorder_profile(result, args.profile)

    return result


def _save_tsv_batch(result: dict, path: str, append: bool = False) -> None:
    """Save or append a TSV row; writes header only when creating the file."""
    import csv

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

    mode = "a" if append else "w"
    write_header = not append or not os.path.exists(path)

    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _run_fasta_entries(entries: list, args: argparse.Namespace) -> None:
    """Run the pipeline on a list of FASTA-parsed protein dicts."""
    print(f"FASTA mode: {len(entries)} protein(s)")
    for i, entry in enumerate(entries):
        label = entry.get("accession") or entry.get("gene_name") or f"protein_{i+1}"
        try:
            # Temporarily patch args.seq so run_single picks up the sequence
            original_seq = getattr(args, "seq", None)
            args.seq = entry["sequence"]
            run_single(label, args, tsv_append=(i > 0))
            args.seq = original_seq
        except Exception as exc:
            print(f"  [SKIP] {label}: {exc}", file=sys.stderr)


def main() -> None:
    args = parse_args()

    # ── FASTA file mode ───────────────────────────────────────────────────
    if getattr(args, "fasta", None):
        if not os.path.isfile(args.fasta):
            print(f"ERROR: FASTA file not found: {args.fasta}", file=sys.stderr)
            sys.exit(1)
        from llps_predictor.fetcher import parse_fasta
        try:
            entries = parse_fasta(args.fasta)
        except Exception as exc:
            print(f"ERROR reading FASTA: {exc}", file=sys.stderr)
            sys.exit(1)
        _run_fasta_entries(entries, args)
        return

    # ── Batch mode (list of IDs) ──────────────────────────────────────────
    if args.batch:
        if not os.path.isfile(args.batch):
            print(f"ERROR: Batch file not found: {args.batch}", file=sys.stderr)
            sys.exit(1)

        with open(args.batch) as f:
            identifiers = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        if not identifiers:
            print("ERROR: No identifiers found in batch file.", file=sys.stderr)
            sys.exit(1)

        print(f"Batch mode: {len(identifiers)} proteins")
        for i, ident in enumerate(identifiers):
            try:
                run_single(ident, args, tsv_append=(i > 0))
            except Exception as exc:
                print(f"  [SKIP] {ident}: {exc}", file=sys.stderr)
        return

    # ── Single mode ───────────────────────────────────────────────────────
    if not args.identifier:
        print(
            "ERROR: Please provide a UniProt accession or gene symbol, "
            "or use --batch / --fasta for multiple proteins.\n\n"
            "Examples:\n"
            "  python main.py TDP43\n"
            "  python main.py Q15653\n"
            "  python main.py FUS --organism human --json fus.json\n"
            "  python main.py MY_PROT --seq MASNDYTQQA...\n"
            "  python main.py --fasta proteins.fasta --tsv results.tsv\n",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        result = run_single(args.identifier, args, tsv_append=False)
        s = result["summary"]
        sys.exit(0 if s["llps_classification"] in ("Low", "Moderate") else 0)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()

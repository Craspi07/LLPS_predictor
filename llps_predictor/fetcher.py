"""
Fetch protein sequences and metadata from UniProt by accession ID or gene symbol.
"""

import time
import requests

UNIPROT_REST = "https://rest.uniprot.org/uniprotkb"

ORGANISM_TAXIDS = {
    "human": 9606,
    "homo sapiens": 9606,
    "mouse": 10090,
    "mus musculus": 10090,
    "rat": 10116,
    "rattus norvegicus": 10116,
    "yeast": 559292,
    "saccharomyces cerevisiae": 559292,
    "ecoli": 83333,
    "e. coli": 83333,
    "escherichia coli": 83333,
    "drosophila": 7227,
    "zebrafish": 7955,
    "worm": 6239,
    "c. elegans": 6239,
}

VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")


_HEADERS = {
    "User-Agent": "LLPS-Predictor/1.0 (github.com/craspi07/llps_predictor; research use)",
    "Accept": "application/json",
}

_NETWORK_HINT = (
    "\nTip: If UniProt is unreachable, provide the sequence directly:\n"
    "  python main.py <label> --seq <AMINO_ACID_SEQUENCE>\n"
    "Or read from a FASTA file:\n"
    "  python main.py <label> --fasta myprotein.fasta\n"
)


def _get(url: str, params: dict = None, retries: int = 4) -> requests.Response:
    """GET with exponential-backoff retry and informative error messages."""
    delay = 2
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=30)
            if r.status_code == 429:
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code in (403, 401):
                raise requests.HTTPError(
                    f"Access denied (HTTP {r.status_code}) for {url}."
                    f"{_NETWORK_HINT}",
                    response=r,
                )
            r.raise_for_status()
            return r
        except requests.ConnectionError as exc:
            last_exc = requests.ConnectionError(
                f"Cannot reach {url}. Check your internet connection.{_NETWORK_HINT}"
            )
            if attempt == retries - 1:
                raise last_exc from exc
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == retries - 1:
                raise
        time.sleep(delay)
        delay *= 2
    raise RuntimeError(f"Failed to GET {url} after {retries} attempts")


def validate_sequence(seq: str) -> str:
    seq = seq.upper().replace(" ", "").replace("\n", "")
    invalid = set(seq) - VALID_AAS
    if invalid:
        raise ValueError(f"Sequence contains invalid amino acid characters: {invalid}")
    if len(seq) < 10:
        raise ValueError("Sequence is too short (< 10 residues)")
    return seq


def _parse_uniprot_entry(entry: dict) -> dict:
    accession = entry.get("primaryAccession", "")
    # sequence
    seq_info = entry.get("sequence", {})
    sequence = seq_info.get("value", "")
    length = seq_info.get("length", len(sequence))
    # protein name
    prot_names = entry.get("proteinDescription", {})
    recommended = prot_names.get("recommendedName", {})
    if recommended:
        protein_name = recommended.get("fullName", {}).get("value", "")
    else:
        # try submitted name
        submitted = prot_names.get("submissionNames", [])
        protein_name = submitted[0].get("fullName", {}).get("value", "") if submitted else ""
    # gene name
    genes = entry.get("genes", [])
    gene_name = ""
    if genes:
        gene_name = genes[0].get("geneName", {}).get("value", "")
    # organism
    organism = entry.get("organism", {}).get("scientificName", "")
    # reviewed
    reviewed = entry.get("entryType", "") == "UniProtKB reviewed (Swiss-Prot)"
    return {
        "accession": accession,
        "gene_name": gene_name,
        "protein_name": protein_name,
        "organism": organism,
        "sequence": sequence,
        "length": length,
        "reviewed": reviewed,
        "source": "UniProt",
    }


def fetch_by_uniprot_id(uniprot_id: str) -> dict:
    """Fetch protein metadata + sequence by UniProt accession (e.g. 'Q15653')."""
    url = f"{UNIPROT_REST}/{uniprot_id.strip()}.json"
    r = _get(url)
    entry = r.json()
    result = _parse_uniprot_entry(entry)
    if not result["sequence"]:
        raise ValueError(f"No sequence found for accession {uniprot_id}")
    result["sequence"] = validate_sequence(result["sequence"])
    return result


def fetch_by_gene_symbol(gene_symbol: str, organism: str = "human") -> dict:
    """
    Search UniProt for a gene symbol and return the best (reviewed) match.
    Prefers Swiss-Prot (reviewed) entries for the given organism.
    """
    taxid = ORGANISM_TAXIDS.get(organism.lower())
    if taxid is None:
        raise ValueError(
            f"Unknown organism '{organism}'. "
            f"Supported: {list(ORGANISM_TAXIDS.keys())}"
        )

    query = f'(gene_exact:{gene_symbol}) AND (organism_id:{taxid}) AND (reviewed:true)'
    params = {
        "query": query,
        "fields": "accession,gene_names,protein_name,organism_name,sequence,reviewed",
        "format": "json",
        "size": 5,
    }
    r = _get(f"{UNIPROT_REST}/search", params=params)
    data = r.json()
    results = data.get("results", [])

    if not results:
        # Broaden to unreviewed
        query_broad = f'(gene_exact:{gene_symbol}) AND (organism_id:{taxid})'
        params["query"] = query_broad
        r2 = _get(f"{UNIPROT_REST}/search", params=params)
        results = r2.json().get("results", [])

    if not results:
        # Last resort: drop gene_exact
        query_loose = f'(gene:{gene_symbol}) AND (organism_id:{taxid}) AND (reviewed:true)'
        params["query"] = query_loose
        r3 = _get(f"{UNIPROT_REST}/search", params=params)
        results = r3.json().get("results", [])

    if not results:
        raise ValueError(
            f"No UniProt entries found for gene '{gene_symbol}' in organism '{organism}' (taxid {taxid})"
        )

    # Pick the reviewed entry with the shortest accession (canonical)
    reviewed = [e for e in results if e.get("entryType", "").startswith("UniProtKB reviewed")]
    candidates = reviewed if reviewed else results
    entry = candidates[0]

    result = _parse_uniprot_entry(entry)
    if not result["sequence"]:
        # Fetch full entry by accession to get sequence
        return fetch_by_uniprot_id(result["accession"])
    result["sequence"] = validate_sequence(result["sequence"])
    return result


def fetch_sequence(identifier: str, organism: str = "human") -> dict:
    """
    Auto-detect whether `identifier` is a UniProt accession or a gene symbol
    and dispatch to the appropriate fetcher.

    UniProt accession pattern: [OPQ][0-9][A-Z0-9]{3}[0-9]  or  [A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}
    Everything else is treated as a gene symbol.
    """
    import re
    uniprot_pattern = re.compile(
        r'^[OPQ][0-9][A-Z0-9]{3}[0-9]$'
        r'|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$',
        re.IGNORECASE,
    )
    ident = identifier.strip()
    if uniprot_pattern.match(ident):
        return fetch_by_uniprot_id(ident)
    else:
        return fetch_by_gene_symbol(ident, organism=organism)


def parse_fasta(fasta_path: str) -> list[dict]:
    """
    Parse a FASTA file and return a list of dicts with keys:
      'accession', 'protein_name', 'sequence', 'length', 'source'

    Handles standard FASTA, UniProt FASTA headers (>sp|ACC|GENE), and
    plain headers (>identifier description).
    """
    import re
    entries = []
    current_header = None
    current_seq_parts = []

    uniprot_header = re.compile(r'^[>]?(sp|tr)\|([A-Z0-9]+)\|(\S+)\s*(.*)')
    plain_header = re.compile(r'^[>]?(\S+)\s*(.*)')

    def _flush():
        if current_header is None:
            return
        seq = validate_sequence("".join(current_seq_parts))
        entries.append({**current_header, "sequence": seq, "length": len(seq), "source": "fasta_file"})

    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                _flush()
                current_seq_parts = []
                m_uni = uniprot_header.match(line)
                if m_uni:
                    current_header = {
                        "accession": m_uni.group(2),
                        "gene_name": m_uni.group(3),
                        "protein_name": m_uni.group(4).strip(),
                        "organism": "",
                        "reviewed": m_uni.group(1) == "sp",
                    }
                else:
                    m_plain = plain_header.match(line)
                    ident = m_plain.group(1) if m_plain else line[1:]
                    desc = m_plain.group(2).strip() if m_plain else ""
                    current_header = {
                        "accession": ident,
                        "gene_name": ident,
                        "protein_name": desc or ident,
                        "organism": "",
                        "reviewed": False,
                    }
            else:
                current_seq_parts.append(line.upper())

    _flush()
    if not entries:
        raise ValueError(f"No valid FASTA entries found in {fasta_path!r}")
    return entries

"""Download every public SARS-CoV-2 Mac1 structure named in the downloaded supplements.

Scans all cells of data/raw/mac1/*.xlsx and the J. Med. Chem. XML for 4-character PDB IDs, keeps
entries whose polymer chain contains a >= 100-residue exact match to the Mac1 sequence of 5SQW
(drops AmpC, sigma-2, D4, Mpro and CHIKV IDs that the supplements also list), and downloads mmCIFs
to data/structures/mac1/. Writes data/processed/mac1_structures.csv, which says for each
structure whether it has an IC50 pair and which split it may be used in.

Run on a login node: python experiments/fetch_mac1_structures.py
"""

import argparse
import json
import re
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

import certifi
import pandas as pd

RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"
RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
REFERENCE_PDB_ID = "5SQW"
MIN_MATCH_RESIDUES = 100
PDB_ID_PATTERN = re.compile(r"[0-9][A-Za-z0-9]{3}")
ENTRY_QUERY = """query($ids:[String!]!){entries(entry_ids:$ids){
  rcsb_id struct{title} exptl{method}
  rcsb_accession_info{initial_release_date}
  rcsb_entry_info{resolution_combined nonpolymer_entity_count}
  polymer_entities{entity_poly{pdbx_seq_one_letter_code_can}}}}"""


def find_pdb_ids(raw_dir: Path) -> dict[str, set[str]]:
    """Candidate 4-character IDs -> supplement files that mention them (all-digit tokens are skipped)."""
    found: dict[str, set[str]] = {}
    for path in sorted(raw_dir.glob("*.xlsx")):
        for sheet in pd.ExcelFile(path).sheet_names:
            for cell in pd.read_excel(path, sheet_name=sheet, header=None, dtype=str).to_numpy().ravel():
                for token in re.split(r"[\s,;/()]+", cell) if isinstance(cell, str) else []:
                    if PDB_ID_PATTERN.fullmatch(token) and not token.isdigit():
                        found.setdefault(token.upper(), set()).add(path.name)
    for path in raw_dir.glob("*.xml"):
        for token in re.findall(r"\b[0-9][A-Z0-9]{3}\b", path.read_text()):
            if not token.isdigit():
                found.setdefault(token, set()).add(path.name)
    return found


def fetch_entries(pdb_ids: list[str]) -> list[dict]:
    """RCSB metadata for the IDs that are public; unreleased IDs come back null and are dropped."""
    entries = []
    for start in range(0, len(pdb_ids), 100):
        payload = json.dumps({"query": ENTRY_QUERY, "variables": {"ids": pdb_ids[start:start + 100]}}).encode()
        request = urllib.request.Request(RCSB_GRAPHQL_URL, payload, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            entries += [entry for entry in json.load(response)["data"]["entries"] if entry]
    return entries


def contains_mac1(entry: dict, reference: str) -> bool:
    for polymer in entry["polymer_entities"]:
        sequence = polymer["entity_poly"]["pdbx_seq_one_letter_code_can"]
        match = SequenceMatcher(None, reference, sequence, autojunk=False).find_longest_match(
            0, len(reference), 0, len(sequence))
        if match.size >= MIN_MATCH_RESIDUES:
            return True
    return False


def download_cif(pdb_id: str, structure_dir: Path) -> None:
    """files.rcsb.org needs certifi: the cluster CA bundle lacks its InCommon intermediate certificate."""
    target = structure_dir / f"{pdb_id}.cif"
    if not target.exists():
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(RCSB_CIF_URL.format(pdb_id=pdb_id), context=context, timeout=120) as response:
            target.write_bytes(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/mac1"))
    parser.add_argument("--structure-dir", type=Path, default=Path("data/structures/mac1"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/mac1_splits.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/mac1_structures.csv"))
    args = parser.parse_args()

    candidates = find_pdb_ids(args.raw_dir)
    entries = fetch_entries(sorted(candidates))
    reference = next(entry for entry in fetch_entries([REFERENCE_PDB_ID])
                     if entry["rcsb_id"] == REFERENCE_PDB_ID)["polymer_entities"][0]["entity_poly"]["pdbx_seq_one_letter_code_can"]
    mac1 = [entry for entry in entries if contains_mac1(entry, reference)]

    args.structure_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda entry: download_cif(entry["rcsb_id"], args.structure_dir), mac1))

    table = pd.DataFrame({
        "pdb_id": [entry["rcsb_id"] for entry in mac1],
        "release_date": [entry["rcsb_accession_info"]["initial_release_date"][:10] for entry in mac1],
        "method": [entry["exptl"][0]["method"] for entry in mac1],
        "resolution_A": [(entry["rcsb_entry_info"]["resolution_combined"] or [None])[0] for entry in mac1],
        "n_nonpolymer_entities": [entry["rcsb_entry_info"]["nonpolymer_entity_count"] for entry in mac1],
        "title": [entry["struct"]["title"] for entry in mac1],
        "mentioned_in": ["; ".join(sorted(candidates[entry["rcsb_id"]])) for entry in mac1],
    })
    splits = pd.read_csv(args.splits).set_index("pdb_id")["split"]
    table["affinity_split"] = table["pdb_id"].map(splits)
    held_out = table["affinity_split"].isin(["val", "test"])
    table["diffusion_train"] = ~held_out
    table = table.sort_values("pdb_id").reset_index(drop=True)
    table.to_csv(args.output, index=False)

    print(f"{len(candidates)} candidate IDs, {len(entries)} public, {len(mac1)} Mac1 structures downloaded to {args.structure_dir}")
    print(table.groupby(table["affinity_split"].fillna("no IC50 pair")).size().rename("structures"))
    print(f"diffusion_train: {table['diffusion_train'].sum()}, held out (val/test): {held_out.sum()}")


if __name__ == "__main__":
    main()

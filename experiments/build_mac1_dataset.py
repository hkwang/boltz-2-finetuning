"""Build a unified Mac1 structure-affinity table from Fraser/Correy lab supplements.

Sources (all SARS-CoV-2 NSP3 macrodomain, Mac1; HTRF peptide-displacement IC50, in uM):
  elife110475  Kim et al., eLife RP110475, Supplementary file 3 (sheet "Mac1")
  scienceadv   Correy et al., Sci. Adv. 2025, ads7187 Table S2 (sheet "B_compounds")
  pnas         Gahbauer et al., PNAS 2023, 2212931120 Dataset S1 (sheet "Compound Summary")
  jmedchem     Jaishankar et al., J. Med. Chem. 2026 (AVI-6451), Tables 1-5 from the PMC full text;
               PDB IDs and ligand SMILES come from RCSB, joined on the AVI compound ID in the entry title
  schuller2021 Schuller et al., Sci. Adv. 2021 (abf8711), Data file S1 sheet "12. solution_binding": the fragments with
               a numeric HTRF IC50 and a PDB ID (manually supplied, data/raw/mac1/abf8711_data_file_s1.xlsx; the ADP-ribose
               control 7KQP is skipped because scienceadv already reports it)
  (eLife 103484, the Mac1 target paper, only reports IC50 in figures, so it adds no rows;
   its structures are already covered by the other sources.)

Run on a login node (needs network). Raw supplements go to data/raw/mac1/, the
table to data/processed/mac1_affinity.csv. Existing raw files are not re-downloaded.
"""

import argparse
import io
import json
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ELIFE_URL = "https://cdn.elifesciences.org/articles/110475/elife-110475-supp{n}-v1.xlsx"
EUROPEPMC_SUPP_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/supplementaryFiles"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"
RCSB_NONPOLYMER_URL = "https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}/{entity_id}"
RCSB_CHEMCOMP_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/{comp_id}"
JMEDCHEM_PMC_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=12788873&rettype=xml"
JMEDCHEM_TABLES = ("T1", "T2", "T3", "T4", "T5")

RAW_FILES = {
    "elife110475_supp3.xlsx": ("elife", 3),
    "elife110475_supp6.xlsx": ("elife", 6),
    "scienceadv_ads7187_table_s2.xlsx": ("europepmc", "PMC12118597", "ads7187_table_s2"),
    "pnas_2212931120_sd01.xlsx": ("europepmc", "PMC9926234", "pnas.2212931120.sd01"),
}
JMEDCHEM_XML = "jmedchem_2026_pmc12788873.xml"
SCHULLER_XLSX = "abf8711_data_file_s1.xlsx"
SCHULLER_CONTROL_PDB = "7KQP"

COLUMNS = [
    "source", "source_compound_id", "pdb_id", "smiles", "ic50_uM", "ic50_qualifier",
    "ic50_raw", "ki_uM", "pdb_in_rcsb", "release_date",
]


def download_raw(raw_dir: Path) -> None:
    """Fetch any missing supplement. Sci. Adv./PNAS ship as nested zips via Europe PMC."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename, spec in RAW_FILES.items():
        target = raw_dir / filename
        if target.exists():
            continue
        if spec[0] == "elife":
            urllib.request.urlretrieve(ELIFE_URL.format(n=spec[1]), target)
            continue
        _, pmcid, stem = spec
        with urllib.request.urlopen(EUROPEPMC_SUPP_URL.format(pmcid=pmcid), timeout=600) as response:
            outer = zipfile.ZipFile(io.BytesIO(response.read()))
        for name in outer.namelist():
            if stem in name and name.endswith((".xlsx", ".zip")):
                payload = outer.read(name)
                if name.endswith(".zip"):
                    inner = zipfile.ZipFile(io.BytesIO(payload))
                    payload = inner.read(next(n for n in inner.namelist() if n.endswith(".xlsx")))
                target.write_bytes(payload)
                break


def download_jmedchem(raw_dir: Path) -> None:
    target = raw_dir / JMEDCHEM_XML
    if not target.exists():
        urllib.request.urlretrieve(JMEDCHEM_PMC_URL, target)


def normalize_avi(compound_id) -> str | None:
    """AVI-0006249 and AVI-6249 are the same compound; return 'AVI-6249'."""
    match = re.fullmatch(r"AVI-0*(\d+)", str(compound_id).strip())
    return f"AVI-{match.group(1)}" if match else None


def rcsb_json(url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload else None
    request = urllib.request.Request(url, data, {"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def fetch_avi_pdb_map() -> pd.DataFrame:
    """All public RCSB Mac1 entries whose title names an AVI compound -> (avi, pdb_id)."""
    query = {
        "query": {"type": "terminal", "service": "full_text",
                  "parameters": {"value": "SARS-CoV-2 NSP3 macrodomain in complex with AVI-"}},
        "return_type": "entry", "request_options": {"return_all_hits": True},
    }
    pdb_ids = [hit["identifier"] for hit in rcsb_json(RCSB_SEARCH_URL, query)["result_set"]]
    graphql = "query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id struct{title}}}"
    rows = []
    for start in range(0, len(pdb_ids), 100):
        batch = rcsb_json(RCSB_GRAPHQL_URL, {"query": graphql, "variables": {"ids": pdb_ids[start:start + 100]}})
        rows += [(normalize_avi(re.search(r"AVI-\d+", entry["struct"]["title"]).group()), entry["rcsb_id"])
                 for entry in batch["data"]["entries"]]
    return pd.DataFrame(rows, columns=["avi", "pdb_id"]).drop_duplicates("avi")


def rcsb_ligand_smiles(pdb_id: str) -> str:
    """SMILES of the largest non-ion ligand deposited in the entry (the modelled ligand)."""
    entry = rcsb_json(RCSB_ENTRY_URL.format(pdb_id=pdb_id))
    best = ("", 0.0)
    for entity_id in entry["rcsb_entry_container_identifiers"].get("non_polymer_entity_ids", []):
        comp_id = rcsb_json(RCSB_NONPOLYMER_URL.format(pdb_id=pdb_id, entity_id=entity_id))["pdbx_entity_nonpoly"]["comp_id"]
        comp = rcsb_json(RCSB_CHEMCOMP_URL.format(comp_id=comp_id))
        weight = comp["chem_comp"].get("formula_weight", 0.0)
        if weight > best[1]:
            best = (comp["rcsb_chem_comp_descriptor"]["SMILES_stereo"], weight)
    return best[0]


def jmedchem_rows(xml_path: Path, avi_to_pdb: dict[str, str]) -> pd.DataFrame:
    """Mac1 HTRF IC50 (uM) from Tables 1-5. A compound in several tables keeps its first value."""
    records = []
    for table in ET.parse(xml_path).getroot().iter("table-wrap"):
        if table.get("id") not in JMEDCHEM_TABLES:
            continue
        cells = [[" ".join("".join(cell.itertext()).split()) for cell in row] for row in table.iter("tr")]
        header = next(index for index, row in enumerate(cells) if row and row[0].lower() == "compound")
        records += [(row[0], row[1]) for row in cells[header + 1:]]
    frame = pd.DataFrame(records, columns=["compound", "ic50"])
    frame["avi"] = frame["compound"].str.extract(r"(AVI-\d+)")[0].map(normalize_avi)
    frame = frame.drop_duplicates("avi")
    frame["pdb_id"] = frame["avi"].map(avi_to_pdb)
    frame = frame.dropna(subset=["pdb_id"])
    frame["smiles"] = frame["pdb_id"].map(rcsb_ligand_smiles)
    return build_rows(frame, "jmedchem", dict(compound_id="avi", pdb_id="pdb_id", smiles="smiles", ic50="ic50"))


def schuller_rows(xlsx_path: Path) -> pd.DataFrame:
    """Fragments in sheet 12 with a numeric IC50 (a leading '~' is dropped) and a PDB ID; salts are stripped to the largest part."""
    sheet = pd.read_excel(xlsx_path, sheet_name="12. solution_binding", header=None)
    sheet = sheet[sheet[20].astype(str).str.fullmatch(r"[0-9][A-Z0-9]{3}") & (sheet[20] != SCHULLER_CONTROL_PDB)].copy()
    sheet["smiles"] = sheet[1].map(lambda smiles: max(smiles.split("."), key=len))
    sheet[16] = sheet[16].astype(str).str.replace("~", "", regex=False)
    sheet = sheet[pd.to_numeric(sheet[16], errors="coerce").notna()]
    return build_rows(sheet, "schuller2021", dict(compound_id=2, pdb_id=20, smiles="smiles", ic50=16))


def parse_ic50(raw) -> tuple[float, str, str]:
    """Return (value_uM, qualifier, raw string). Multi-valued cells (isomer pairs) give NaN."""
    if pd.isna(raw):
        return float("nan"), "", ""
    text = str(raw).strip()
    match = re.fullmatch(r"([<>]=?)?\s*([0-9.]+(?:[eE][+-]?[0-9]+)?)", text)
    if match is None:
        return float("nan"), "", text
    return float(match.group(2)), match.group(1) or "=", text


def build_rows(frame: pd.DataFrame, source: str, columns: dict) -> pd.DataFrame:
    frame = frame.dropna(subset=[columns["pdb_id"], columns["smiles"]]).copy()
    parsed = frame[columns["ic50"]].map(parse_ic50)
    out = pd.DataFrame({
        "source": source,
        "source_compound_id": frame[columns["compound_id"]].astype(str),
        "pdb_id": frame[columns["pdb_id"]].astype(str).str.split().str[0].str.upper(),
        "smiles": frame[columns["smiles"]],
        "ic50_uM": parsed.map(lambda value: value[0]),
        "ic50_qualifier": parsed.map(lambda value: value[1]),
        "ic50_raw": parsed.map(lambda value: value[2]),
        "ki_uM": pd.to_numeric(frame[columns["ki"]], errors="coerce") if columns.get("ki") else float("nan"),
    })
    return out


def fetch_release_dates(pdb_ids: list[str]) -> dict[str, str | None]:
    """RCSB initial release date (YYYY-MM-DD), or None if the entry is not public (404)."""
    def release_date(pdb_id: str) -> str | None:
        try:
            entry = rcsb_json(RCSB_ENTRY_URL.format(pdb_id=pdb_id))
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise
        return entry["rcsb_accession_info"]["initial_release_date"][:10]

    with ThreadPoolExecutor(16) as pool:
        return dict(zip(pdb_ids, pool.map(release_date, pdb_ids)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/mac1"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/mac1_affinity.csv"))
    args = parser.parse_args()

    download_raw(args.raw_dir)
    download_jmedchem(args.raw_dir)
    avi_pdb = fetch_avi_pdb_map()
    avi_to_pdb = dict(zip(avi_pdb["avi"], avi_pdb["pdb_id"]))

    elife = pd.read_excel(args.raw_dir / "elife110475_supp3.xlsx", sheet_name="Mac1")
    scienceadv = pd.read_excel(args.raw_dir / "scienceadv_ads7187_table_s2.xlsx", sheet_name="B_compounds", header=1)
    pnas = pd.read_excel(args.raw_dir / "pnas_2212931120_sd01.xlsx", sheet_name="Compound Summary")

    for frame, avi_column, pdb_column in [(elife, "AVI_ID", "PDB ID"), (scienceadv, "AVI-ID", "PDB code")]:
        frame[pdb_column] = frame[pdb_column].fillna(frame[avi_column].map(normalize_avi).map(avi_to_pdb))
    scienceadv["smiles_used"] = scienceadv["SMILES_nosalt"].fillna(scienceadv["SMILES"])
    table = pd.concat([
        build_rows(elife, "elife110475", dict(
            compound_id="AVI_ID", pdb_id="PDB ID", smiles="SMILES", ic50="IC50 (uM)")),
        build_rows(scienceadv, "scienceadv", dict(
            compound_id="AVI-ID", pdb_id="PDB code", smiles="smiles_used", ic50="IC50 (μM)", ki="Ki (μM)")),
        build_rows(pnas, "pnas", dict(
            compound_id="ID", pdb_id="PDB", smiles="Smiles", ic50="IC50 [µM]", ki="Ki*** [µM]")),
        jmedchem_rows(args.raw_dir / JMEDCHEM_XML, avi_to_pdb),
        schuller_rows(args.raw_dir / SCHULLER_XLSX),
    ], ignore_index=True)

    release_dates = fetch_release_dates(sorted(table["pdb_id"].unique()))
    table["release_date"] = table["pdb_id"].map(release_dates)
    table["pdb_in_rcsb"] = table["release_date"].notna()
    table = table[COLUMNS].sort_values(["source", "pdb_id"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    has_ic50 = table["ic50_uM"].notna()
    print(table.groupby("source").agg(
        rows=("pdb_id", "size"), unique_pdb=("pdb_id", "nunique"),
        with_ic50=("ic50_uM", lambda column: column.notna().sum()),
        pdb_in_rcsb=("pdb_in_rcsb", "sum"),
        usable=("pdb_in_rcsb", lambda column: (column & has_ic50[column.index]).sum()),
    ))
    print(f"wrote {len(table)} rows to {args.output}")


if __name__ == "__main__":
    main()

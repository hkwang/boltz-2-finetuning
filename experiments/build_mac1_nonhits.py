"""Build a Mac1 non-hit table from Schuller et al., Sci. Adv. 2021 (abf8711), Data file S1.

Tiers (column `evidence`):
  xtal_nonhit               fragment soaked into Mac1 crystals (P43 and/or C2), no ligand density
                            (sheets "8. P43_all_datasets", "10. C2_all_datasets"). Weak negative: soak
                            concentration, crystal packing and loop conformation limit sensitivity.
  docked_unconfirmed_htrf   docked compound that was not confirmed by soaking and was tested by DSF/HTRF
                            (sheet "12. solution_binding"). `pct_competition` is the HTRF peptide
                            displacement at `htrf_top_conc_mM`; it is not thresholded here.

Fragments that are a hit in any crystal form, or that appear in data/processed/mac1_affinity.csv, are dropped
(counted in the printout) so a compound is never both a hit and a non-hit. The input is the local
data/raw/mac1/abf8711_data_file_s1.xlsx; no network is needed.
"""

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

SUPP = "abf8711_data_file_s1.xlsx"
SOLUTION_UNCONFIRMED = "Docked compounds not confirmed by soaking"
SOLUTION_NEXT_SECTION = "XChem screening hits"
COLUMNS = [
    "evidence", "compound_id", "library", "smiles", "crystal_forms", "soak_time_hrs", "pandda_analyzed",
    "dsf_dTm_C", "pct_competition", "htrf_top_conc_mM", "source",
]


def flat_smiles(smiles: str) -> str | None:
    """Canonical SMILES of the largest fragment without stereo (salts dropped), used only to match molecules across tables."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    largest = max(Chem.GetMolFrags(mol, asMols=True), key=lambda fragment: fragment.GetNumHeavyAtoms())
    return Chem.MolToSmiles(largest, isomericSmiles=False)


def crystal_screen_rows(path: Path) -> tuple[pd.DataFrame, set[str]]:
    """Non-hit rows from the P43 and C2 sheets (merged per molecule) and the set of hit molecules."""
    p43 = pd.read_excel(path, "8. P43_all_datasets").assign(crystal_form="P43")
    c2 = pd.read_excel(path, "10. C2_all_datasets").assign(crystal_form="C2")
    screens = pd.concat([p43, c2], ignore_index=True)
    screens["flat"] = screens["Smiles"].map(flat_smiles)
    hits = set(screens.loc[screens["Fragment?"], "flat"])
    nonhits = screens[~screens["Fragment?"] & ~screens["flat"].isin(hits) & screens["flat"].notna()]
    rows = []
    for flat, group in nonhits.groupby("flat", sort=False):
        first = group.iloc[0]
        rows.append({
            "evidence": "xtal_nonhit",
            "compound_id": ";".join(dict.fromkeys(group["Compound_ID"].astype(str))),
            "library": ";".join(dict.fromkeys(group["Library"].astype(str))),
            "smiles": first["Smiles"],
            "crystal_forms": ";".join(sorted(set(group["crystal_form"]))),
            "soak_time_hrs": group["Soak_time_hrs"].max(),
            "pandda_analyzed": bool(group["PanDDA?"].all()),
            "source": "abf8711_data_file_s1:sheets 8,10",
            "flat": flat,
        })
    return pd.DataFrame(rows), hits


def unconfirmed_docked_rows(path: Path) -> pd.DataFrame:
    """Docked compounds not confirmed by soaking, with DSF dTm and HTRF % competition."""
    sheet = pd.read_excel(path, "12. solution_binding", header=None)
    start = sheet.index[sheet[0] == SOLUTION_UNCONFIRMED][0] + 1
    stop = sheet.index[sheet[0] == SOLUTION_NEXT_SECTION][0]
    block = sheet.loc[start : stop - 1]
    block = block[block[1].notna()]
    return pd.DataFrame({
        "evidence": "docked_unconfirmed_htrf",
        "compound_id": block[2].fillna(block[0]).astype(str),
        "library": block[0].astype(str),
        "smiles": block[1],
        "dsf_dTm_C": pd.to_numeric(block[7], errors="coerce"),
        "pct_competition": pd.to_numeric(block[18], errors="coerce"),
        "htrf_top_conc_mM": pd.to_numeric(block[19], errors="coerce"),
        "source": "abf8711_data_file_s1:sheet 12",
        "flat": block[1].map(flat_smiles),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/mac1"))
    parser.add_argument("--affinity", type=Path, default=Path("data/processed/mac1_affinity.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/mac1_nonhits.csv"))
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")

    path = args.raw_dir / SUPP
    xtal, crystal_hits = crystal_screen_rows(path)
    docked = unconfirmed_docked_rows(path)
    table = pd.concat([xtal, docked], ignore_index=True)

    affinity_flat = set(pd.read_csv(args.affinity)["smiles"].map(flat_smiles))
    known_binder = table["flat"].isin(affinity_flat | crystal_hits)
    print(f"dropped {known_binder.sum()} rows matching a hit or a mac1_affinity.csv molecule")
    table = table[~known_binder].drop_duplicates(["evidence", "flat"]).reindex(columns=COLUMNS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.groupby("evidence").size().to_string())
    print(f"wrote {len(table)} rows to {args.output}")


if __name__ == "__main__":
    main()

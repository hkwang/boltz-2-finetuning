"""Assign each IC50 pair its ligand component code(s) by comparing the table SMILES to the mmCIF ligands.

The table SMILES are vendor/design SMILES; the structure's `_pdbx_entity_nonpoly` lists the deposited components.
Both sides are reduced to InChIKeys: a 'stereo' match agrees on connectivity and stereochemistry, a 'connectivity'
match only on the skeleton (the table SMILES is racemic/unspecified or drawn with the other enantiomer), and 'none'
means the deposited ligand is a different compound. Several candidates remain when a structure models both
enantiomers of a racemic compound. Component CIFs missing from Boltz's CCD cache are downloaded to data/raw/ccd/.

Run on a login node: python experiments/match_mac1_ligands.py
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd

from boltz2_finetuning.ligands import (
    assign_ligand,
    component_inchikey,
    download_component,
    model_smiles,
    nonpolymer_codes,
    smiles_inchikey,
)

CCD_CACHE = Path.home() / ".cache" / "boltz" / "mols"
BUILT_MOLS = Path("data/processed/mac1_mols")


def load_component_mol(code: str):
    for directory in (BUILT_MOLS, CCD_CACHE):
        if (directory / f"{code}.pkl").exists():
            return pickle.load((directory / f"{code}.pkl").open("rb"))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--affinity", type=Path, default=Path("data/processed/mac1_affinity.csv"))
    parser.add_argument("--structure-dir", type=Path, default=Path("data/structures/mac1"))
    parser.add_argument("--component-dir", type=Path, default=Path("data/raw/ccd"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/mac1_ligands.csv"))
    args = parser.parse_args()
    args.component_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.affinity)
    pairs = table[table["pdb_in_rcsb"] & table["ic50_uM"].notna()]

    component_keys: dict[str, str | None] = {}
    rows = []
    for pair in pairs.itertuples():
        codes = nonpolymer_codes(args.structure_dir / f"{pair.pdb_id}.cif")
        for code in codes:
            if code not in component_keys:
                component_keys[code] = component_inchikey(download_component(code, args.component_dir))
        table_key = smiles_inchikey(pair.smiles)
        keys = {code: component_keys[code] for code in codes if component_keys[code]}
        matched, level = assign_ligand(table_key, keys) if table_key else ([], "none")
        modelled = []
        for code in matched:
            mol = load_component_mol(code)
            smiles, consistent = model_smiles(args.structure_dir / f"{pair.pdb_id}.cif", code, mol) if mol else (None, True)
            modelled.append((smiles or "", smiles is not None and smiles_inchikey(smiles) == keys[code], consistent))
        rows.append({
            "source": pair.source,
            "source_compound_id": pair.source_compound_id,
            "pdb_id": pair.pdb_id,
            "smiles": pair.smiles,
            "nonpolymer_codes": " ".join(codes),
            "ligand_codes": " ".join(matched),
            "match": level,
            "model_smiles": " ".join(smiles for smiles, _, _ in modelled),
            "model_stereo_agrees_with_component": all(agrees for _, agrees, _ in modelled) if modelled else False,
            "model_copies_consistent": all(consistent for _, _, consistent in modelled),
            "in_boltz_ccd": all((CCD_CACHE / f"{code}.pkl").exists() for code in matched) if matched else False,
        })
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    result["n_candidates"] = result["ligand_codes"].str.split().str.len().fillna(0).astype(int)
    print(f"{len(result)} pairs -> {args.output}")
    print(result["match"].value_counts().to_string())
    print("candidates per pair:", result["n_candidates"].value_counts().sort_index().to_dict())
    print("model stereo disagrees with component:", int((~result["model_stereo_agrees_with_component"] & (result["match"] != "none")).sum()))
    print("copies disagree within a structure:", int((~result["model_copies_consistent"]).sum()))
    print("ligand missing from Boltz CCD cache:", int((~result["in_boltz_ccd"] & (result["match"] != "none")).sum()))
    print(result[result["match"] == "none"][["source_compound_id", "pdb_id", "nonpolymer_codes", "smiles"]].to_string())


if __name__ == "__main__":
    main()

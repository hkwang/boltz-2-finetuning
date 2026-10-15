"""Convert the Mac1 structures and fragment-screen negatives into Boltz-2 training inputs.

Writes data/processed/boltz_inputs/{structures,constraints,mols,records}/ plus manifest.json (all records), and
data/processed/mac1_training_records.csv, which says how each record may be used:
  kind             crystal (real pose, from data/structures/mac1/{pdb_id}.cif) or nonhit (SMILES only, apo 7KQO sequence)
  diffusion_train  crystal records only; False for val/test entries (mac1_structures.csv)
  affinity_kind    ic50 (measured, mac1_splits.csv), binary (fragment-hit adenosine-site positive or screen negative) or none
  affinity_split   train / val / test / held_out_similar for the affinity head; empty when affinity_kind is none
Affinity ligand: the first deposited copy of the first matched ligand code (mac1_ligands.csv); the 29 racemic pairs with
two enantiomer codes therefore use one of them. 7HFU and 7HFX (table SMILES != deposited ligand) get no affinity label.
Binary positives come from mac1_fragment_hit_records.csv in the chain stated there; an IC50 label wins over a binary one.

Run: PYTHONPATH=src python experiments/build_mac1_boltz_inputs.py
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from boltz.data.mol import load_canonicals
from boltz.data.types import Manifest
from rdkit.Chem import Descriptors  # noqa: F401  (Boltz's schema parser calls AllChem.Descriptors without importing it)

from boltz2_finetuning.records import convert_crystal, convert_smiles, write_inputs

CCD_CACHE = Path.home() / ".cache" / "boltz" / "mols"
COLUMNS = [
    "record_id", "kind", "pdb_id", "diffusion_train", "structure_loss_weight", "affinity_kind", "affinity_split",
    "affinity_code", "affinity_chain", "log10_ic50_uM", "ic50_qualifier", "affinity_binary", "label_quality", "affinity_note",
]


def crystal_labels(splits: pd.DataFrame, ligands: pd.DataFrame, hits: pd.DataFrame) -> dict[str, dict]:
    """pdb_id -> affinity label fields (IC50 pairs first, then binary fragment-hit positives)."""
    labels = {}
    for pair in splits.itertuples():
        candidates = ligands[(ligands["pdb_id"] == pair.pdb_id) & (ligands["smiles"] == pair.smiles)]
        matched = candidates[candidates["match"] != "none"]
        if matched.empty:
            labels[pair.pdb_id] = {"affinity_kind": "none", "affinity_note": "table SMILES does not match the deposited ligand"}
            continue
        labels[pair.pdb_id] = {
            "affinity_kind": "ic50",
            "affinity_split": pair.split,
            "affinity_code": matched.iloc[0]["ligand_codes"].split()[0],
            "log10_ic50_uM": float(np.log10(pair.ic50_uM)),
            "ic50_qualifier": pair.ic50_qualifier,
            "label_quality": "measured",
        }
    for hit in hits[hits["affinity_binary"] == 1].itertuples():
        if hit.pdb_id not in labels:
            labels[hit.pdb_id] = {
                "affinity_kind": "binary",
                "affinity_split": hit.split,
                "affinity_code": hit.ligand_code,
                "affinity_chain": hit.chain,
                "affinity_binary": 1,
                "label_quality": hit.label_quality,
            }
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--structure-dir", type=Path, default=Path("data/structures/mac1"))
    parser.add_argument("--mol-dir", type=Path, default=Path("data/processed/mac1_mols"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/boltz_inputs"))
    parser.add_argument("--table", type=Path, default=Path("data/processed/mac1_training_records.csv"))
    parser.add_argument("--limit", type=int, help="convert only the first N crystal and N negative records (testing)")
    args = parser.parse_args()

    processed = Path("data/processed")
    structures = pd.read_csv(processed / "mac1_structures.csv")
    labels = crystal_labels(
        pd.read_csv(processed / "mac1_splits.csv"),
        pd.read_csv(processed / "mac1_ligands.csv").fillna(""),
        pd.read_csv(processed / "mac1_fragment_hit_records.csv"),
    )
    negatives = pd.read_csv(processed / "mac1_nonhit_records.csv")
    if args.limit:
        structures, negatives = structures.head(args.limit), negatives.head(args.limit)

    built_mols = {path.stem: pickle.load(path.open("rb")) for path in args.mol_dir.glob("*.pkl")}
    canonicals = load_canonicals(str(CCD_CACHE))
    rows, records, failures = [], [], []

    for entry in structures.itertuples():
        label = labels.get(entry.pdb_id, {"affinity_kind": "none"})
        try:
            record, structure, constraints, extra_mols = convert_crystal(
                entry.pdb_id,
                args.structure_dir / f"{entry.pdb_id}.cif",
                built_mols,
                CCD_CACHE,
                label.get("affinity_code"),
                label.get("affinity_chain"),
            )
        except Exception as error:
            failures.append((entry.pdb_id, repr(error)))
            continue
        write_inputs(args.out_dir, record, structure, constraints, extra_mols)
        records.append(record)
        rows.append({
            "record_id": entry.pdb_id, "kind": "crystal", "pdb_id": entry.pdb_id,
            "diffusion_train": bool(entry.diffusion_train), "structure_loss_weight": 1.0 if entry.diffusion_train else 0.0,
            "affinity_note": label.get("affinity_note"), **{key: value for key, value in label.items() if key != "affinity_note"},
        })

    for negative in negatives.itertuples():
        try:
            record, structure, constraints, extra_mols = convert_smiles(
                negative.record_id, negative.sequence, negative.smiles, canonicals, CCD_CACHE
            )
        except Exception as error:
            failures.append((negative.record_id, repr(error)))
            continue
        write_inputs(args.out_dir, record, structure, constraints, extra_mols)
        records.append(record)
        rows.append({
            "record_id": negative.record_id, "kind": "nonhit", "pdb_id": negative.pdb_id, "diffusion_train": False,
            "structure_loss_weight": 0.0, "affinity_kind": "binary", "affinity_split": negative.split,
            "affinity_binary": 0, "label_quality": negative.label_quality,
        })

    Manifest(records=records).dump(args.out_dir / "manifest.json")
    table = pd.DataFrame(rows).reindex(columns=COLUMNS)
    table.to_csv(args.table, index=False)
    print(f"wrote {len(records)} records to {args.out_dir} and {args.table}")
    print(table.groupby(["kind", "affinity_kind"], dropna=False).size().to_string())
    print(f"{len(failures)} failures")
    for record_id, error in failures:
        print("FAILED", record_id, error)


if __name__ == "__main__":
    main()

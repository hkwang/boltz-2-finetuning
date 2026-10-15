"""Write one Boltz-2 input YAML per held-out (val + test) Mac1 entry that has a measured or censored IC50 and a deposited pose.

Run from the repository root: python experiments/prepare_mac1_eval.py
The ligand SMILES is the one re-derived from the deposited model (`model_smiles` in mac1_ligands.csv) for the labelled ligand
code, so its stereochemistry matches the crystal pose. The protein is the construct and `msa: empty` of configs/mac1_cofold.yaml.
Also writes eval_entries.csv: the label, split and SMILES for each entry (read by experiments/evaluate_mac1.py).
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cofold-config", type=Path, default=Path("configs/mac1_cofold.yaml"))
    parser.add_argument("--records", type=Path, default=Path("data/processed/mac1_training_records.csv"))
    parser.add_argument("--ligands", type=Path, default=Path("data/processed/mac1_ligands.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/mac1_eval"))
    args = parser.parse_args()

    cofold = yaml.safe_load(args.cofold_config.read_text())
    records = pd.read_csv(args.records)
    held_out = records[(records["kind"] == "crystal") & records["affinity_split"].isin(["val", "test"]) & (records["affinity_kind"] == "ic50")]
    ligands = pd.read_csv(args.ligands).drop_duplicates("pdb_id").set_index("pdb_id")
    inputs_dir = args.out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for entry in held_out.itertuples():
        codes = ligands.loc[entry.record_id, "ligand_codes"].split()
        smiles = ligands.loc[entry.record_id, "model_smiles"].split()[codes.index(entry.affinity_code)]
        boltz_input = {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": cofold["protein_sequence"], "msa": cofold["msa"]}},
                {"ligand": {"id": "B", "smiles": smiles}},
            ],
            "properties": [{"affinity": {"binder": "B"}}],
        }
        (inputs_dir / f"mac1_eval_{entry.record_id}.yaml").write_text(yaml.safe_dump(boltz_input, sort_keys=False))
        rows.append(
            {
                "pdb_id": entry.record_id,
                "split": entry.affinity_split,
                "ligand_code": entry.affinity_code,
                "smiles": smiles,
                "ic50_qualifier": entry.ic50_qualifier,
                "log10_ic50_uM": entry.log10_ic50_uM,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "eval_entries.csv", index=False)
    print(f"wrote {len(table)} inputs to {inputs_dir}: {table['split'].value_counts().to_dict()}, censored {(table['ic50_qualifier'] == '>').sum()}")


if __name__ == "__main__":
    main()

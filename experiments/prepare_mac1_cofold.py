"""Write one Boltz-2 input YAML per ligand listed in configs/mac1_cofold.yaml.

Run from the repository root: python experiments/prepare_mac1_cofold.py
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/mac1_cofold.yaml"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/mac1_splits.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/mac1_cofold/inputs"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    pairs = pd.read_csv(args.splits).set_index("pdb_id")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for pdb_id in config["ligands"]:
        pair = pairs.loc[pdb_id]
        boltz_input = {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": config["protein_sequence"], "msa": config["msa"]}},
                {"ligand": {"id": "B", "smiles": pair["smiles"]}},
            ],
            "properties": [{"affinity": {"binder": "B"}}],
        }
        path = args.out_dir / f"{config['name']}_{pdb_id}.yaml"
        path.write_text(yaml.safe_dump(boltz_input, sort_keys=False))
        print(f"{path}  split={pair['split']}  IC50 {pair['ic50_qualifier']} {pair['ic50_uM']} uM")


if __name__ == "__main__":
    main()

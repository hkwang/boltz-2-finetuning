"""Collect Boltz-2 Mac1 co-folding outputs next to the experimental IC50 labels.

Run from the repository root: python experiments/summarize_mac1_cofold.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/mac1_cofold.yaml"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/mac1_splits.csv"))
    parser.add_argument("--results", type=Path, default=Path("experiments/results/mac1_cofold"))
    args = parser.parse_args()

    name = yaml.safe_load(args.config.read_text())["name"]
    labels = pd.read_csv(args.splits).set_index("pdb_id")
    rows = []
    for pdb_id in yaml.safe_load(args.config.read_text())["ligands"]:
        prediction_dir = args.results / "boltz_results_inputs" / "predictions" / f"{name}_{pdb_id}"
        confidence = json.loads(next(prediction_dir.glob("confidence_*_model_0.json")).read_text())
        affinity = json.loads((prediction_dir / f"affinity_{name}_{pdb_id}.json").read_text())
        rows.append(
            {
                "pdb_id": pdb_id,
                "ic50_qualifier": labels.loc[pdb_id, "ic50_qualifier"],
                "ic50_uM": labels.loc[pdb_id, "ic50_uM"],
                "boltz_affinity_pred_value": affinity["affinity_pred_value"],
                "boltz_affinity_probability_binary": affinity["affinity_probability_binary"],
                "confidence_score": confidence["confidence_score"],
                "iptm": confidence["iptm"],
                "ligand_iptm": confidence["ligand_iptm"],
                "structure": str(next(prediction_dir.glob("*_model_0.cif"))),
            }
        )
    table = pd.DataFrame(rows)
    output = args.results / "summary.csv"
    table.to_csv(output, index=False)
    print(table.drop(columns="structure").to_string(index=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

"""Score Boltz-2 co-folding predictions on the held-out Mac1 entries: ligand pose RMSD and affinity vs experimental IC50.

Run: python experiments/evaluate_mac1.py RESULTS_DIR [--label NAME]
RESULTS_DIR is a `boltz predict` output directory (containing boltz_results_*/predictions/mac1_eval_<PDB>/) made from the YAMLs of
experiments/prepare_mac1_eval.py. Writes RESULTS_DIR/eval_per_entry.csv and RESULTS_DIR/eval_summary.json.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from boltz2_finetuning.evaluation import affinity_summary, load_component, pose_metrics, pose_summary, read_prediction

INPUTS_DIR = Path("data/processed/boltz_inputs")
CCD_DIR = Path.home() / ".cache" / "boltz" / "mols"
STRUCTURE_DIR = Path("data/structures/mac1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--entries", type=Path, default=Path("data/processed/mac1_eval/eval_entries.csv"))
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    prediction_root = next(args.results.glob("boltz_results_*")) / "predictions"
    rows = []
    for entry in pd.read_csv(args.entries).itertuples():
        name = f"mac1_eval_{entry.pdb_id}"
        directory = prediction_root / name
        row = {"pdb_id": entry.pdb_id, "split": entry.split, "ic50_qualifier": entry.ic50_qualifier, "log10_ic50_uM": entry.log10_ic50_uM}
        samples = read_prediction(directory, name) if directory.exists() else []
        if not samples:
            rows.append({**row, "error": "no prediction"})
            continue
        component = load_component(entry.pdb_id, entry.ligand_code, INPUTS_DIR, CCD_DIR)
        poses = []
        for sample in samples:
            try:
                metrics = pose_metrics(directory / f"{name}_model_{sample['model']}.cif", STRUCTURE_DIR / f"{entry.pdb_id}.cif", entry.ligand_code, component, entry.smiles)
                poses.append({**metrics, "model": sample["model"], "confidence_score": sample["confidence_score"], "iptm": sample["iptm"]})
            except ValueError as error:
                row["error"] = str(error)
        affinity_path = directory / f"affinity_{name}.json"
        affinity = json.loads(affinity_path.read_text()) if affinity_path.exists() else {}
        if poses:
            row.update(
                rmsd_top1=poses[0]["ligand_rmsd"],
                rmsd_best=min(pose["ligand_rmsd"] for pose in poses),
                n_samples=len(poses),
                pocket_ca_rmsd=poses[0]["pocket_ca_rmsd"],
                ca_rmsd=poses[0]["ca_rmsd"],
                deposited_occupancy=poses[0]["deposited_occupancy"],
                confidence_score=poses[0]["confidence_score"],
                iptm=poses[0]["iptm"],
            )
        row.update(affinity_pred_value=affinity.get("affinity_pred_value"), affinity_probability_binary=affinity.get("affinity_probability_binary"))
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(args.results / "eval_per_entry.csv", index=False)
    scored = table.dropna(subset=["rmsd_top1"]) if "rmsd_top1" in table else table.iloc[0:0]
    summary = {
        "label": args.label,
        "n_entries": int(len(table)),
        "n_pose_scored": int(len(scored)),
        "pose": pose_summary(scored) if len(scored) else None,
        "affinity": affinity_summary(table) if table["affinity_pred_value"].notna().sum() > 2 else None,
    }
    for split in ("val", "test"):
        part = scored[scored["split"] == split]
        summary[f"pose_{split}"] = pose_summary(part) if len(part) > 2 else None
    (args.results / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

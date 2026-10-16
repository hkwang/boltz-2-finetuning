"""Rescore stored Boltz-2 validation predictions per diffusion sample (all 5), for metrics beyond top-1 and best-of-5.

Run: python experiments/rescore_samples.py OUT.csv RESULTS_DIR [RESULTS_DIR ...]
Each RESULTS_DIR is a `boltz predict` output (as made by experiments/run_mac1_eval.sh). Uses the frozen scoring functions in
boltz2_finetuning.evaluation unchanged and scores only the validation entries. Writes one row per (results dir, entry, sample):
run, pdb_id, rank (0 = top confidence), model, ligand_rmsd, confidence_score, iptm, error. Unreadable directories are skipped.
"""

import argparse
from pathlib import Path

import pandas as pd
from boltz2_finetuning.evaluation import load_component, pose_metrics, read_prediction

INPUTS_DIR = Path("data/processed/boltz_inputs")
CCD_DIR = Path.home() / ".cache" / "boltz" / "mols"
STRUCTURE_DIR = Path("data/structures/mac1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--entries", type=Path, default=Path("data/processed/mac1_eval/eval_entries.csv"))
    args = parser.parse_args()

    entries = pd.read_csv(args.entries)
    entries = entries[entries["split"] == "val"]
    components = {entry.pdb_id: load_component(entry.pdb_id, entry.ligand_code, INPUTS_DIR, CCD_DIR) for entry in entries.itertuples()}
    rows = []
    for results in args.results:
        roots = list(results.glob("boltz_results_*/predictions"))
        if not roots:
            print(f"skip {results}: no predictions")
            continue
        for entry in entries.itertuples():
            name = f"mac1_eval_{entry.pdb_id}"
            directory = roots[0] / name
            samples = read_prediction(directory, name) if directory.exists() else []
            for rank, sample in enumerate(samples):
                row = {"run": str(results), "pdb_id": entry.pdb_id, "rank": rank, "model": sample["model"],
                       "confidence_score": sample["confidence_score"], "iptm": sample["iptm"]}
                try:
                    metrics = pose_metrics(directory / f"{name}_model_{sample['model']}.cif", STRUCTURE_DIR / f"{entry.pdb_id}.cif",
                                           entry.ligand_code, components[entry.pdb_id], entry.smiles)
                    row["ligand_rmsd"] = metrics["ligand_rmsd"]
                except ValueError as error:
                    row["error"] = str(error)
                rows.append(row)
        print(f"scored {results}", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

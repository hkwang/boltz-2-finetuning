"""Summarize per-sample rescoring tables (experiments/rescore_samples.py) into one per-run, per-entry table.

Run: python experiments/summarize_rescore.py RESCORE_DIR OUT.csv
For every RESCORE_DIR/<agent>__<results dir name>.csv writes one row per (agent, results dir, entry): top1, best, median, mean,
worst, n_samples and frac_samples_below_2A of the per-sample ligand RMSD. Also checks top1/best against eval_per_entry.csv
when the results directory is given with --check and prints the largest mismatch.
"""

import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    rescore, out = Path(sys.argv[1]), Path(sys.argv[2])
    parts = []
    for path in sorted(rescore.glob("*.csv")):
        table = pd.read_csv(path)
        if table.empty or "ligand_rmsd" not in table:
            continue
        agent, run = path.stem.split("__", 1)
        grouped = table.dropna(subset=["ligand_rmsd"]).sort_values("rank").groupby("pdb_id")["ligand_rmsd"]
        summary = pd.DataFrame({
            "top1": grouped.first(), "best": grouped.min(), "median": grouped.median(), "mean": grouped.mean(), "worst": grouped.max(),
            "n_samples": grouped.size(), "frac_samples_below_2A": grouped.apply(lambda values: float((values < 2.0).mean())),
        }).reset_index()
        summary.insert(0, "run", run)
        summary.insert(0, "agent", agent)
        summary["results_dir"] = table["run"].iloc[0]
        parts.append(summary)
    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(out, index=False)
    mismatch = []
    for results_dir, part in combined.groupby("results_dir"):
        reference = Path(results_dir) / "eval_per_entry.csv"
        if reference.exists():
            frozen = pd.read_csv(reference).set_index("pdb_id")
            joined = part.set_index("pdb_id").join(frozen[["rmsd_top1", "rmsd_best"]], how="inner")
            mismatch.append(max((joined["top1"] - joined["rmsd_top1"]).abs().max(), (joined["best"] - joined["rmsd_best"]).abs().max()))
    print(f"{len(combined)} rows, {combined['results_dir'].nunique()} runs; checked {len(mismatch)} against eval_per_entry.csv, max |diff| {max(mismatch) if mismatch else float('nan'):.2e}")


if __name__ == "__main__":
    main()

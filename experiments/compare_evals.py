"""Table of held-out Mac1 metrics for evaluation runs under experiments/results/mac1_eval/.

Run: python experiments/compare_evals.py [LABEL ...]   (default: stock plus every directory with an eval_summary.json)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("experiments/results/mac1_eval")


def matching_stock(label: str) -> str:
    """Stock run with the same predict seed as `label` (labels ending `_seed<N>` use seed N; others use seed 0)."""
    suffix = label.rsplit("_", 1)[-1]
    return f"stock_{suffix}" if suffix.startswith("seed") and label != f"stock_{suffix}" else "stock"


def paired_difference(label: str, baseline: str, resamples: int = 5000) -> tuple[float, float, float]:
    """Median per-entry change in top-1 ligand RMSD (label minus baseline, A) with a bootstrap 95% CI over entries."""
    both = pd.read_csv(ROOT / label / "eval_per_entry.csv").merge(
        pd.read_csv(ROOT / baseline / "eval_per_entry.csv"), on="pdb_id", suffixes=("", "_base")).dropna(subset=["rmsd_top1", "rmsd_top1_base"])
    delta = (both["rmsd_top1"] - both["rmsd_top1_base"]).to_numpy()
    generator = np.random.default_rng(0)
    medians = [np.median(generator.choice(delta, len(delta))) for _ in range(resamples)]
    return float(np.median(delta)), float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def main() -> None:
    labels = sys.argv[1:] or sorted(path.parent.name for path in ROOT.glob("*/eval_summary.json"))
    print(f"{'label':28} {'n':>3} {'rmsd_top1':>9} {'<2A':>5} {'<1A':>5} {'rmsd_best':>9} {'spearman':>8} {'rmse':>6} {'bias':>6}  {'paired d(top1) vs same-seed stock [95% CI]':>34}")
    for label in sorted(labels, key=lambda name: (name != "stock", name)):
        summary = json.loads((ROOT / label / "eval_summary.json").read_text())
        if not summary["affinity"] or summary["n_pose_scored"] < 40:
            continue
        top1, best, affinity = summary["pose"]["rmsd_top1"], summary["pose"]["rmsd_best"], summary["affinity"]
        delta = "" if label in ("stock", "stock_seed1", "stock_seed2") else "%+.2f [%+.2f, %+.2f]" % paired_difference(label, matching_stock(label))
        print(f"{label:28} {summary['n_pose_scored']:>3} {top1['median']:>9.2f} {top1['frac_below_2A']:>5.2f} {top1['frac_below_1A']:>5.2f} "
              f"{best['median']:>9.2f} {affinity['spearman']:>8.2f} {affinity['rmse_log10']:>6.2f} {affinity['mean_bias_log10']:>6.2f}  {delta}")


if __name__ == "__main__":
    main()

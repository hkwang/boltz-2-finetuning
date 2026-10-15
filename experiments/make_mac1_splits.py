"""Deduplicate the Mac1 table and write leakage-aware train/val/test splits.

Run from the repository root: PYTHONPATH=src python experiments/make_mac1_splits.py
"""

import argparse
from pathlib import Path

import pandas as pd

from boltz2_finetuning.splits import assign_splits, deduplicate_by_pdb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=Path("data/processed/mac1_affinity.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/mac1_splits.csv"))
    parser.add_argument("--cutoff-date", default="2023-06-01", help="Boltz-2 training cutoff (eLife 110475 Supp. 6)")
    parser.add_argument("--tanimoto-threshold", type=float, default=0.5)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pairs = deduplicate_by_pdb(pd.read_csv(args.input))
    pairs = assign_splits(
        pairs, args.cutoff_date, args.tanimoto_threshold, args.val_fraction, args.test_fraction, args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output, index=False)

    print(f"args: {vars(args)}")
    summary = pairs.groupby("split").agg(
        pairs=("pdb_id", "size"),
        clusters=("cluster", "nunique"),
        pre_cutoff=("pre_cutoff", "sum"),
        censored=("ic50_qualifier", lambda column: (column != "=").sum()),
        median_ic50_uM=("ic50_uM", "median"),
    )
    print(summary)
    print(f"wrote {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()

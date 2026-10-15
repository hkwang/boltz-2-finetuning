"""Validation-only summary of one candidate in the docs/results_reporting.md format, from its per-seed eval_per_entry.csv tables.

Usage: python experiments/report_val.py DIR [--label NAME]
DIR holds one subdirectory per predict seed (seed2026/, seed1/, seed2/ as written by swarm/iri.py post-result), each with eval_per_entry.csv.
Prints, for median rmsd_best (best-of-5, the reporting headline) and median rmsd_top1 (the leaderboard score), the mean over seeds with the
over-seeds t-interval (n = number of seeds) and, per seed, the over-entries bootstrap interval (2000 resamples). Test rows are dropped first.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BOOTSTRAPS = 2000


def bootstrap_median(values: np.ndarray, seed: int = 0) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    medians = [np.median(generator.choice(values, len(values))) for _ in range(BOOTSTRAPS)]
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def seed_interval(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    half = stats.t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values))
    return float(np.mean(values) - half), float(np.mean(values) + half)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    tables = {}
    for path in sorted(args.directory.glob("seed*/eval_per_entry.csv")):
        table = pd.read_csv(path)
        tables[int(path.parent.name.removeprefix("seed"))] = table[table["split"] == "val"]
    if not tables:
        raise SystemExit(f"no seed*/eval_per_entry.csv under {args.directory}")
    print(f"{args.label or args.directory.name}: validation only, {len(next(iter(tables.values())))} entries, seeds {sorted(tables)}")
    for metric in ("rmsd_best", "rmsd_top1"):
        per_seed = {seed: float(table[metric].median()) for seed, table in tables.items()}
        low, high = seed_interval(list(per_seed.values()))
        print(f"  median {metric}: {np.mean(list(per_seed.values())):.3f} A, over-seeds 95% t-interval [{low:.3f}, {high:.3f}]")
        for seed, table in sorted(tables.items()):
            b_low, b_high = bootstrap_median(table[metric].dropna().to_numpy())
            print(f"    seed {seed}: {per_seed[seed]:.3f} A, over-entries bootstrap 95% [{b_low:.3f}, {b_high:.3f}]")


if __name__ == "__main__":
    main()

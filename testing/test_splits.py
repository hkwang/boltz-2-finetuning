"""Checks the split invariants on the real Mac1 table. Run: PYTHONPATH=src python testing/test_splits.py"""

import pandas as pd

from boltz2_finetuning.splits import assign_splits, deduplicate_by_pdb

CUTOFF = "2023-06-01"


def build_splits(seed: int = 0) -> pd.DataFrame:
    pairs = deduplicate_by_pdb(pd.read_csv("data/processed/mac1_affinity.csv"))
    return assign_splits(pairs, CUTOFF, 0.5, 0.15, 0.15, seed)


def test_one_row_per_pdb_and_measured_preferred() -> None:
    table = pd.read_csv("data/processed/mac1_affinity.csv")
    pairs = deduplicate_by_pdb(table)
    assert pairs["pdb_id"].is_unique
    measured = table[table["pdb_in_rcsb"] & (table["ic50_qualifier"] == "=")]["pdb_id"]
    censored_kept = pairs[pairs["ic50_qualifier"] != "="]["pdb_id"]
    assert not censored_kept.isin(measured).any()


def test_no_leakage_into_held_out_sets() -> None:
    splits = build_splits()
    held_out = splits[splits["split"] != "train"]
    assert len(held_out) > 0
    assert not held_out["pre_cutoff"].any()
    pre_cutoff_clusters = set(splits.loc[splits["pre_cutoff"], "cluster"])
    assert not held_out["cluster"].isin(pre_cutoff_clusters).any()


def test_clusters_not_split_across_sets() -> None:
    splits = build_splits()
    assert (splits.groupby("cluster")["split"].nunique() == 1).all()


def test_splits_depend_only_on_seed() -> None:
    assert build_splits(0).equals(build_splits(0))
    assert not build_splits(0)["split"].equals(build_splits(1)["split"])


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok {name}")

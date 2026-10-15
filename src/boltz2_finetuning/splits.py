"""Deduplication and leakage-aware train/val/test splits for structure-affinity pairs."""

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

SOURCE_PRIORITY = ["jmedchem", "scienceadv", "pnas", "elife110475", "schuller2021"]


def deduplicate_by_pdb(table: pd.DataFrame) -> pd.DataFrame:
    """Collapse the build table to one row per PDB entry with a measured, public IC50.

    A structure is labelled once, so conflicting sources must be resolved. Preference: a measured
    ('=') value over a censored ('>') one, then SOURCE_PRIORITY (jmedchem SMILES come from the
    deposited ligand). The IC50 spread across sources is kept so conflicts stay visible.
    """
    usable = table[table["pdb_in_rcsb"] & table["ic50_uM"].notna()].copy()
    usable["censored"] = usable["ic50_qualifier"] != "="
    usable["priority"] = usable["source"].map(SOURCE_PRIORITY.index)
    usable = usable.sort_values(["pdb_id", "censored", "priority"])
    grouped = usable.groupby("pdb_id")
    deduplicated = grouped.first()
    deduplicated["sources"] = grouped["source"].agg(lambda sources: "+".join(sorted(set(sources))))
    deduplicated["ic50_min_uM"] = grouped["ic50_uM"].min()
    deduplicated["ic50_max_uM"] = grouped["ic50_uM"].max()
    columns = ["smiles", "ic50_uM", "ic50_qualifier", "sources", "ic50_min_uM", "ic50_max_uM", "release_date"]
    return deduplicated[columns].reset_index()


def cluster_ligands(smiles: list[str], tanimoto_threshold: float) -> np.ndarray:
    """Butina clusters on Morgan (r=2, 2048 bit) Tanimoto distance; members are >= threshold similar to a centroid."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [generator.GetFingerprint(Chem.MolFromSmiles(smi)) for smi in smiles]
    distances = []
    for index in range(1, len(fingerprints)):
        distances += [1 - sim for sim in DataStructs.BulkTanimotoSimilarity(fingerprints[index], fingerprints[:index])]
    clusters = Butina.ClusterData(distances, len(fingerprints), 1 - tanimoto_threshold, isDistData=True)
    labels = np.empty(len(fingerprints), dtype=int)
    for label, members in enumerate(clusters):
        labels[list(members)] = label
    return labels


def assign_splits(
    pairs: pd.DataFrame,
    cutoff_date: str,
    tanimoto_threshold: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Add `pre_cutoff`, `cluster` and `split` (train/val/test) columns.

    - Structures released on or before `cutoff_date` may be in Boltz-2 pretraining: train only.
    - Ligands are clustered; a cluster is never split across sets.
    - Any cluster containing a pre-cutoff structure is train only, so val/test contain neither
      pretraining structures nor ligands close to them.
    - The remaining clusters are shuffled (seeded) and filled into test, then val, up to their
      target sizes; everything else is train.
    """
    pairs = pairs.copy()
    pairs["pre_cutoff"] = pairs["release_date"] <= cutoff_date
    pairs["cluster"] = cluster_ligands(pairs["smiles"].tolist(), tanimoto_threshold)
    train_only = set(pairs.loc[pairs["pre_cutoff"], "cluster"])
    held_out_candidates = [c for c in pairs["cluster"].unique() if c not in train_only]
    np.random.default_rng(seed).shuffle(held_out_candidates)

    sizes = pairs["cluster"].value_counts()
    targets = {"test": test_fraction * len(pairs), "val": val_fraction * len(pairs)}
    filled = {"test": 0, "val": 0}
    cluster_split = {}
    for cluster in held_out_candidates:
        for name in ("test", "val"):
            if filled[name] + sizes[cluster] <= targets[name]:
                cluster_split[cluster] = name
                filled[name] += sizes[cluster]
                break
    pairs["split"] = pairs["cluster"].map(cluster_split).fillna("train")
    return pairs


def close_to_held_out(smiles: pd.Series, splits: pd.DataFrame, tanimoto_threshold: float) -> pd.Series:
    """True where a ligand has Morgan (r=2, 2048 bit) Tanimoto >= threshold to any val/test ligand in `splits`."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    held_out = [generator.GetFingerprint(Chem.MolFromSmiles(value)) for value in splits.loc[splits["split"] != "train", "smiles"]]
    return smiles.map(
        lambda value: max(DataStructs.BulkTanimotoSimilarity(generator.GetFingerprint(Chem.MolFromSmiles(value)), held_out))
        >= tanimoto_threshold
    )

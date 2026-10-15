"""Write one Boltz-2 input YAML per affinity-training record and the label table used by experiments/train_affinity.py.

Run from the repository root: python experiments/prepare_affinity_inputs.py [--shards 3]
Rows (train split only; val/test IC50 entries are the held-out set of experiments/prepare_mac1_eval.py):
  ic50    measured or censored IC50 pairs (SMILES re-derived from the deposited model, as in prepare_mac1_eval.py)
  hit     adenosine-site fragment hits, binary label 1
  nonhit  fragment-screen negatives, binary label 0 (weak negatives, lower sample weight)
Binary labels of ic50 rows: 1 if the measured IC50 is at most BINDER_MAX_UM, else 0 (censored `>` rows, and measured values above it).
YAMLs are written to data/processed/mac1_affinity/shard<k>/ so each shard can be run by one `boltz predict` job to generate poses.
Outputs: yamls in shard dirs and data/processed/mac1_affinity/labels.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BINDER_MAX_UM = 100.0
SAMPLE_WEIGHT = {"measured": 1.0, "xtal_hit": 1.0, "weak_htrf": 0.5, "weak": 0.2}


def build_rows(processed: Path, cofold_sequence: str) -> pd.DataFrame:
    records = pd.read_csv(processed / "mac1_training_records.csv")
    splits = pd.read_csv(processed / "mac1_splits.csv").set_index("pdb_id")
    ligands = pd.read_csv(processed / "mac1_ligands.csv").fillna("").drop_duplicates("pdb_id").set_index("pdb_id")
    hits = pd.read_csv(processed / "mac1_fragment_hit_records.csv")
    nonhits = pd.read_csv(processed / "mac1_nonhit_records.csv")
    rows = []

    for entry in records[(records["affinity_kind"] == "ic50") & (records["affinity_split"] == "train")].itertuples():
        codes = ligands.loc[entry.record_id, "ligand_codes"].split()
        smiles = ligands.loc[entry.record_id, "model_smiles"].split()[codes.index(entry.affinity_code)]
        censored = entry.ic50_qualifier == ">"
        ic50_uM = 10 ** entry.log10_ic50_uM
        rows.append(dict(
            record_id=entry.record_id, kind="ic50", sequence=cofold_sequence, smiles=smiles, assay_id=splits.loc[entry.record_id, "sources"],
            log10_ic50_uM=entry.log10_ic50_uM, censored=censored, binary=int((not censored) and ic50_uM <= BINDER_MAX_UM),
            weight=SAMPLE_WEIGHT["measured"],
        ))
    chosen = records[(records["affinity_kind"] == "binary") & (records["kind"] == "crystal") & (records["affinity_split"] == "train")]
    for entry in chosen.itertuples():
        hit = hits[(hits["pdb_id"] == entry.record_id) & (hits["chain"] == entry.affinity_chain)].iloc[0]
        rows.append(dict(
            record_id=entry.record_id, kind="hit", sequence=hit["sequence"], smiles=hit["smiles"], assay_id="abf8711_hits",
            log10_ic50_uM=np.nan, censored=False, binary=1, weight=SAMPLE_WEIGHT[hit["label_quality"]],
        ))
    for entry in nonhits.itertuples():
        rows.append(dict(
            record_id=entry.record_id, kind="nonhit", sequence=entry.sequence, smiles=entry.smiles, assay_id="abf8711_nonhits",
            log10_ic50_uM=np.nan, censored=False, binary=0, weight=SAMPLE_WEIGHT[entry.label_quality],
        ))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/mac1_affinity"))
    args = parser.parse_args()

    cofold = yaml.safe_load(Path("configs/mac1_cofold.yaml").read_text())
    table = build_rows(Path("data/processed"), cofold["protein_sequence"])
    tables = {name: part for name, part in table.groupby("kind")}
    table["shard"] = np.concatenate([np.arange(len(part)) % args.shards for part in tables.values()])[np.argsort(np.concatenate([part.index for part in tables.values()]))]
    for row in table.itertuples():
        directory = args.out_dir / f"shard{row.shard}"
        directory.mkdir(parents=True, exist_ok=True)
        boltz_input = {
            "version": 1,
            "sequences": [{"protein": {"id": "A", "sequence": row.sequence, "msa": "empty"}}, {"ligand": {"id": "B", "smiles": row.smiles}}],
            "properties": [{"affinity": {"binder": "B"}}],
        }
        (directory / f"{row.record_id}.yaml").write_text(yaml.safe_dump(boltz_input, sort_keys=False))
    table.to_csv(args.out_dir / "labels.csv", index=False)
    print(table.groupby("kind").agg(n=("record_id", "size"), binary=("binary", "sum"), censored=("censored", "sum")).to_string())
    print(table.groupby("shard").size().to_dict())


if __name__ == "__main__":
    main()

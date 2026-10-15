"""Format Mac1 crystallographic non-hits as training records: {sequence, ligand SMILES, apo P43 structure}.

Each record pairs one `xtal_nonhit` ligand (data/processed/mac1_nonhits.csv) with the apo P43 Mac1 structure
7KQO, chain A (protomer A is where ligands were observed in the fragment screen; the adenosine site of
protomer B is blocked by crystal packing). The structure is referenced by `pdb_id` like the other tables
(data/structures/mac1/{pdb_id}.cif, listed in mac1_structures.csv). There is no ligand pose, so
`structure_loss_weight` is 0 and the only supervision is `affinity_binary` = 0. `label_quality` = weak: a
crystallographic non-hit is not a measured non-binder. Docked compounds that were not confirmed by soaking are
included as `label_quality` = weak_htrf when their HTRF peptide displacement at the top concentration is below
`--max-pct-competition` (default 30%); those with no HTRF value or more displacement are not clear negatives and are
left out. `split` is train for all rows: none is >= 0.5 Tanimoto
(Morgan r=2, 2048 bit) to a val/test ligand in mac1_splits.csv.

Run from the repository root: PYTHONPATH=src python experiments/build_mac1_nonhit_records.py
"""

import argparse
from pathlib import Path

import gemmi
import pandas as pd
import yaml
from rdkit import RDLogger

from boltz2_finetuning.splits import close_to_held_out

RDLogger.DisableLog("rdApp.*")
CHAIN = "A"
HELD_OUT_SIMILARITY = 0.5


def chain_sequence(structure_path: Path, chain_name: str) -> str:
    structure = gemmi.read_structure(str(structure_path))
    chain = structure[0][chain_name]
    return gemmi.one_letter_code([residue.name for residue in chain.get_polymer()])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nonhits", type=Path, default=Path("data/processed/mac1_nonhits.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/mac1_cofold.yaml"))
    parser.add_argument("--pdb-id", default="7KQO")
    parser.add_argument("--max-pct-competition", type=float, default=30.0)
    parser.add_argument("--structure-dir", type=Path, default=Path("data/structures/mac1"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/mac1_splits.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/mac1_nonhit_records.csv"))
    args = parser.parse_args()

    sequence = yaml.safe_load(args.config.read_text())["protein_sequence"]
    assert chain_sequence(args.structure_dir / f"{args.pdb_id}.cif", CHAIN) == sequence, "structure chain does not match protein_sequence"

    nonhits = pd.read_csv(args.nonhits)
    docked_negative = (nonhits["evidence"] == "docked_unconfirmed_htrf") & (nonhits["pct_competition"] < args.max_pct_competition)
    print(f"docked compounds kept as negatives: {docked_negative.sum()} of {(nonhits['evidence'] == 'docked_unconfirmed_htrf').sum()}")
    nonhits = nonhits[(nonhits["evidence"] == "xtal_nonhit") | docked_negative].reset_index(drop=True)
    leaking = close_to_held_out(nonhits["smiles"], pd.read_csv(args.splits), HELD_OUT_SIMILARITY)
    print(f"dropped {leaking.sum()} non-hits >= {HELD_OUT_SIMILARITY} Tanimoto to a val/test ligand")
    nonhits = nonhits[~leaking].reset_index(drop=True)
    records = pd.DataFrame({
        "record_id": [f"mac1_nonhit_{index:04d}" for index in nonhits.index],
        "sequence": sequence,
        "smiles": nonhits["smiles"],
        "pdb_id": args.pdb_id,
        "chain": CHAIN,
        "split": "train",
        "structure_loss_weight": 0.0,
        "affinity_binary": 0,
        "label_quality": nonhits["evidence"].map({"xtal_nonhit": "weak", "docked_unconfirmed_htrf": "weak_htrf"}),
        "compound_id": nonhits["compound_id"],
        "library": nonhits["library"],
        "source": nonhits["source"],
    })
    print(records["label_quality"].value_counts().to_string())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.output, index=False)
    print(f"wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()

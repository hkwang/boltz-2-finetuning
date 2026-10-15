"""Checks the converted Boltz inputs against the label tables. Run: PYTHONPATH=src python testing/test_boltz_inputs.py
(after experiments/build_mac1_boltz_inputs.py). Featurizes a seeded sample of records with Boltz-2's own tokenizer/featurizer."""

from pathlib import Path

import numpy as np
import pandas as pd
from boltz.data.mol import load_canonicals, load_molecules
from boltz.data.module.inferencev2 import load_input
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.types import Manifest, Record, StructureV2
from rdkit.Chem import Descriptors  # noqa: F401

ROOT = Path("data/processed/boltz_inputs")
CCD_CACHE = Path.home() / ".cache" / "boltz" / "mols"
TABLE = pd.read_csv("data/processed/mac1_training_records.csv")
SAMPLE_SIZE = 40


def test_manifest_matches_table_and_files() -> None:
    manifest = Manifest.load(ROOT / "manifest.json")
    assert sorted(record.id for record in manifest.records) == sorted(TABLE["record_id"])
    for record_id in TABLE["record_id"]:
        for folder, suffix in (("structures", "npz"), ("constraints", "npz"), ("mols", "pkl"), ("records", "json")):
            assert (ROOT / folder / f"{record_id}.{suffix}").exists(), (folder, record_id)


def test_affinity_ligand_is_the_labelled_one() -> None:
    labelled = TABLE[TABLE["affinity_kind"] != "none"]
    for row in labelled.itertuples():
        record = Record.load(ROOT / "records" / f"{row.record_id}.json")
        assert record.affinity is not None, row.record_id
        structure = StructureV2.load(ROOT / "structures" / f"{row.record_id}.npz")
        chain = structure.chains[record.affinity.chain_id]
        assert chain["mol_type"] == 3 and chain["res_num"] == 1, row.record_id
        name = str(structure.residues[chain["res_idx"]]["name"])
        if row.kind == "crystal":
            assert name == row.affinity_code, (row.record_id, name, row.affinity_code)
        assert 50 < record.affinity.mw < 700, (row.record_id, record.affinity.mw)
    unlabelled = TABLE[TABLE["affinity_kind"] == "none"]
    assert all(Record.load(ROOT / "records" / f"{rid}.json").affinity is None for rid in unlabelled["record_id"])


def test_held_out_never_trains_the_diffusion_head() -> None:
    crystals = TABLE[TABLE["kind"] == "crystal"]
    held_out = crystals[crystals["affinity_split"].isin(["val", "test"])]
    assert len(held_out) > 0 and not held_out["diffusion_train"].any()
    assert (TABLE.loc[~TABLE["diffusion_train"], "structure_loss_weight"] == 0).all()
    assert (TABLE.loc[TABLE["kind"] == "nonhit", "structure_loss_weight"] == 0).all()


def test_sample_featurizes_with_boltz2() -> None:
    canonicals = load_canonicals(str(CCD_CACHE))
    tokenizer, featurizer = Boltz2Tokenizer(), Boltz2Featurizer()
    sample = TABLE.sample(SAMPLE_SIZE, random_state=0)
    for row in sample.itertuples():
        record = Record.load(ROOT / "records" / f"{row.record_id}.json")
        data = load_input(record, ROOT / "structures", Path("/nonexistent"), constraints_dir=ROOT / "constraints", extra_mols_dir=ROOT / "mols")
        tokenized = tokenizer.tokenize(data)
        molecules = {**canonicals, **data.extra_mols}
        molecules.update(load_molecules(str(CCD_CACHE), set(tokenized.tokens["res_name"].tolist()) - set(molecules)))
        features = featurizer.process(
            tokenized, molecules=molecules, random=np.random.default_rng(0), training=True, max_atoms=None, max_tokens=None,
            max_seqs=1, pad_to_max_seqs=False, single_sequence_prop=0.0, compute_frames=True,
            compute_constraint_features=True, compute_symmetries=True, compute_affinity=False,
        )
        expected = 0 if row.affinity_kind == "none" else 1
        assert (features["affinity_token_mask"].sum() > 0) == bool(expected), row.record_id


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)

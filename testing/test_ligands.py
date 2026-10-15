"""Checks the ligand cross-reference. Run: PYTHONPATH=src python testing/test_ligands.py (after experiments/match_mac1_ligands.py)"""

import pandas as pd

from boltz2_finetuning.ligands import assign_ligand, match_level

LIGANDS = pd.read_csv("data/processed/mac1_ligands.csv").fillna("")


def test_match_levels() -> None:
    assert match_level("AAAAAAAAAAAAAA-BBBBBBBBCC-N", "AAAAAAAAAAAAAA-BBBBBBBBCC-N") == "stereo"
    assert match_level("AAAAAAAAAAAAAA-BBBBBBBBCC-N", "AAAAAAAAAAAAAA-DDDDDDDDCC-N") == "connectivity"
    assert match_level("AAAAAAAAAAAAAA-BBBBBBBBCC-N", "ZZZZZZZZZZZZZZ-BBBBBBBBCC-N") == "none"


def test_stereo_match_beats_connectivity_match() -> None:
    keys = {"AAA": "AAAAAAAAAAAAAA-DDDDDDDDCC-N", "BBB": "AAAAAAAAAAAAAA-BBBBBBBBCC-N"}
    assert assign_ligand("AAAAAAAAAAAAAA-BBBBBBBBCC-N", keys) == (["BBB"], "stereo")


def test_every_assigned_code_is_deposited_in_its_structure() -> None:
    for row in LIGANDS[LIGANDS["match"] != "none"].itertuples():
        assert set(row.ligand_codes.split()) <= set(row.nonpolymer_codes.split())


def test_only_known_mismatches_have_no_ligand() -> None:
    assert set(LIGANDS.loc[LIGANDS["match"] == "none", "pdb_id"]) == {"7HFU", "7HFX"}


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)

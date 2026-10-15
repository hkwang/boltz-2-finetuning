"""Checks the regression scoring. Run: PYTHONPATH=src python testing/test_regression.py (needs data/structures/regression/*.cif)"""

from pathlib import Path

import gemmi
import numpy as np
import torch

from boltz2_finetuning.regression import (
    compare_to_reference,
    kabsch,
    polymer_sequence,
    read_structure,
    reference_ca,
    reference_ligand,
    write_finetuned_checkpoint,
)

rng = np.random.default_rng(0)


def random_rotation() -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def load_complex(pdb_id: str, ligand: str) -> tuple[dict, dict]:
    structure = read_structure(Path(f"data/structures/regression/{pdb_id}.cif"))
    return reference_ca(structure, "A"), reference_ligand(structure, "A", ligand)


def move(coords: dict, rotation: np.ndarray, shift: np.ndarray) -> dict:
    return {key: value @ rotation + shift for key, value in coords.items()}


def test_kabsch_recovers_rigid_motion() -> None:
    points = rng.normal(size=(30, 3))
    moved = points @ random_rotation() + 5.0
    rotation, shift = kabsch(moved, points)
    assert np.allclose(moved @ rotation + shift, points, atol=1e-8)


def test_rigidly_moved_copy_scores_zero() -> None:
    ca, ligand = load_complex("1AQ1", "STU")
    rotation, shift = random_rotation(), np.array([10.0, -4.0, 7.0])
    scores = compare_to_reference(ca, ligand, move(ca, rotation, shift), move(ligand, rotation, shift))
    assert scores["ca_rmsd"] < 1e-6 and scores["ligand_rmsd_global"] < 1e-6 and scores["ligand_rmsd_pocket"] < 1e-6
    assert scores["n_pocket_residues"] > 10


def test_displaced_ligand_is_detected_in_both_frames() -> None:
    ca, ligand = load_complex("3RY2", "BTN")
    displaced = {name: xyz + np.array([3.0, 0.0, 0.0]) for name, xyz in ligand.items()}
    scores = compare_to_reference(ca, ligand, ca, displaced)
    assert scores["ca_rmsd"] < 1e-6
    assert np.isclose(scores["ligand_rmsd_global"], 3.0) and np.isclose(scores["ligand_rmsd_pocket"], 3.0)


def test_missing_residues_are_ignored_not_misaligned() -> None:
    ca, ligand = load_complex("1IEP", "STI")
    partial = {index: xyz for index, xyz in ca.items() if index % 2}
    scores = compare_to_reference(ca, ligand, partial, ligand)
    assert scores["n_residues"] == len(partial) and scores["ca_rmsd"] < 1e-6


def test_sequences_match_observed_residues() -> None:
    for pdb_id in ("3RY2", "1AQ1", "1IEP"):
        structure = read_structure(Path(f"data/structures/regression/{pdb_id}.cif"))
        sequence = polymer_sequence(structure, "A")
        observed = reference_ca(structure, "A")
        assert min(observed) >= 1 and max(observed) <= len(sequence)
        for residue in structure[0]["A"]:
            if residue.het_flag == "A" and residue.label_seq is not None:
                assert sequence[residue.label_seq - 1] == gemmi.find_tabulated_residue(residue.name).one_letter_code.upper()


def test_finetuned_checkpoint_replaces_only_listed_tensors(tmp_path: Path = Path("/tmp/regression_test")) -> None:
    tmp_path.mkdir(exist_ok=True)
    base = {"state_dict": {"a.weight": torch.zeros(2), "b.weight": torch.ones(2)}, "hyper_parameters": {"x": 1}}
    torch.save(base, tmp_path / "base.ckpt")
    torch.save({"a.weight": torch.full((2,), 5.0)}, tmp_path / "weights.pt")
    assert write_finetuned_checkpoint(tmp_path / "base.ckpt", tmp_path / "weights.pt", tmp_path / "out.ckpt") == 1
    out = torch.load(tmp_path / "out.ckpt", weights_only=False)
    assert torch.equal(out["state_dict"]["a.weight"], torch.full((2,), 5.0))
    assert torch.equal(out["state_dict"]["b.weight"], torch.ones(2)) and out["hyper_parameters"] == {"x": 1}
    torch.save({"a.weight": torch.zeros(2)}, tmp_path / "same.pt")
    try:
        write_finetuned_checkpoint(tmp_path / "base.ckpt", tmp_path / "same.pt", tmp_path / "out2.ckpt")
    except AssertionError:
        return
    raise AssertionError("identical weights must be rejected")


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)

"""Checks the RMSD/superposition math. Run: PYTHONPATH=src python testing/test_evaluation.py"""

import numpy as np
import gemmi
from boltz2_finetuning.evaluation import kabsch, ligand_copies, pose_summary, rmsd
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd


def random_rotation(generator: np.random.Generator) -> np.ndarray:
    matrix, _ = np.linalg.qr(generator.normal(size=(3, 3)))
    return matrix * np.sign(np.linalg.det(matrix))


def test_kabsch_recovers_rigid_motion() -> None:
    generator = np.random.default_rng(0)
    fixed = generator.normal(size=(30, 3)) * 5
    rotation = random_rotation(generator)
    moving = fixed @ rotation.T + np.array([3.0, -2.0, 7.0])
    found_rotation, translation = kabsch(moving, fixed)
    assert rmsd(moving @ found_rotation.T + translation, fixed) < 1e-9
    assert np.linalg.det(found_rotation) > 0


def test_kabsch_never_reflects() -> None:
    generator = np.random.default_rng(1)
    fixed = generator.normal(size=(20, 3))
    mirrored = fixed * np.array([1.0, 1.0, -1.0])
    rotation, _ = kabsch(mirrored, fixed)
    assert np.linalg.det(rotation) > 0
    assert rmsd(mirrored @ rotation.T, fixed) > 0.1


def test_symmetry_matches_give_zero_for_swapped_equivalent_atoms() -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles("Oc1ccc(F)cc1"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    heavy = Chem.RemoveHs(mol)
    coords = heavy.GetConformer().GetPositions()
    matches = heavy.GetSubstructMatches(Chem.MolFromSmiles("Oc1ccc(F)cc1"), uniquify=False)
    assert len(matches) == 2
    flipped = coords[list(matches[1])]
    assert min(rmsd(flipped, coords[list(match)]) for match in matches) < 1e-9
    assert rmsd(flipped, coords) > 0.1


def test_pose_summary_thresholds() -> None:
    table = pd.DataFrame({"rmsd_top1": [0.5, 1.5, 3.0, 8.0], "rmsd_best": [0.4, 0.9, 2.5, 6.0]})
    summary = pose_summary(table)
    assert summary["rmsd_top1"]["frac_below_2A"] == 0.5
    assert summary["rmsd_best"]["frac_below_1A"] == 0.5


def test_ligand_copies_split_altlocs_and_skip_hydrogens() -> None:
    structure = gemmi.Structure()
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    residue = gemmi.Residue()
    residue.name = "LIG"
    residue.seqid = gemmi.SeqId(1, " ")
    rows = [("C1", "\0", 1.0, "C", 0.0), ("N1", "A", 0.7, "N", 1.0), ("N1", "B", 0.3, "N", 5.0), ("H1", "\0", 1.0, "H", 9.0)]
    for name, altloc, occupancy, element, x_value in rows:
        atom = gemmi.Atom()
        atom.name = name
        atom.altloc = altloc
        atom.occ = occupancy
        atom.element = gemmi.Element(element)
        atom.pos = gemmi.Position(x_value, 0.0, 0.0)
        residue.add_atom(atom)
    chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    copies = ligand_copies(structure, "LIG", ["C1", "N1"])
    assert len(copies) == 2
    assert {round(float(coordinates[1, 0])) for coordinates, _ in copies} == {1, 5}
    assert sorted(round(occupancy, 2) for _, occupancy in copies) == [0.65, 0.85]


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)

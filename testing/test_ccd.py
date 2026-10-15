"""Compares rebuilt molecules with Boltz's cached ones. Run: PYTHONPATH=src python testing/test_ccd.py"""

import pickle
from pathlib import Path

import numpy as np

from boltz2_finetuning.ccd import build_ccd_mol

CCD_CACHE = Path.home() / ".cache" / "boltz" / "mols"
COMPONENT_DIR = Path("data/raw/ccd")
CODES = ["QW3", "QZF", "R0R", "RWQ", "A1AJW", "DMS"]


def unpack(mol, name: str):
    return pickle.loads(bytes.fromhex(mol.GetProp(name)))


def test_matches_cached_molecules() -> None:
    for code in CODES:
        cached = pickle.load((CCD_CACHE / f"{code}.pkl").open("rb"))
        built = pickle.loads(pickle.dumps(build_ccd_mol(code, COMPONENT_DIR / f"{code}.cif")))
        for field in ("name", "alt_name", "leaving_atom"):
            assert [a.GetProp(field) for a in cached.GetAtoms()] == [a.GetProp(field) for a in built.GetAtoms()], (code, field)
        assert [a.GetAtomicNum() for a in cached.GetAtoms()] == [a.GetAtomicNum() for a in built.GetAtoms()]
        assert [a.GetFormalCharge() for a in cached.GetAtoms()] == [a.GetFormalCharge() for a in built.GetAtoms()]
        assert [str(a.GetChiralTag()) for a in cached.GetAtoms()] == [str(a.GetChiralTag()) for a in built.GetAtoms()]
        bonds = lambda mol: sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType().name) for b in mol.GetBonds())
        assert bonds(cached) == bonds(built), code
        assert set(unpack(cached, "symmetries")) == set(unpack(built, "symmetries")), code
        assert built.GetNumConformers() == cached.GetNumConformers() == 10
        reference = lambda mol: set(map(tuple, unpack(mol, "chiral_atom_index").T[unpack(mol, "chiral_check_mask")]))
        assert reference(cached) == reference(built), code
        assert np.array_equal(unpack(cached, "aromatic_6_ring_index"), unpack(built, "aromatic_6_ring_index")), code


def test_conformer_has_ideal_bond_lengths() -> None:
    mol = build_ccd_mol("QW3", COMPONENT_DIR / "QW3.cif")
    positions = mol.GetConformer(0).GetPositions()
    lengths = [np.linalg.norm(positions[b.GetBeginAtomIdx()] - positions[b.GetEndAtomIdx()]) for b in mol.GetBonds()]
    assert 1.0 < min(lengths) and max(lengths) < 1.8


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)

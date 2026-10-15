"""Build Boltz-style CCD molecules from RCSB chemical component CIFs.

Boltz reads ligands from ~/.cache/boltz/mols/{CODE}.pkl, but that cache lacks the newest 5-character A1xxx codes.
`build_ccd_mol` reproduces the cache format: heavy atoms in CIF order with `name`/`alt_name`/`leaving_atom` atom
properties, 10 ETKDG conformers, and hex-pickled `symmetries` and constraint arrays computed with Boltz's own
functions (the same ones `parse_ccd_residue` uses), so the constraints match the cached molecules.
"""

import pickle
from pathlib import Path

import gemmi
import numpy as np
from boltz.data.parse.schema import (
    compute_chiral_atom_constraints,
    compute_flatness_constraints,
    compute_geometry_constraints,
    compute_stereo_bond_constraints,
)
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

NUM_CONFORMERS = 10
MAX_SYMMETRIES = 1000
BOND_ORDERS = {"SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE, "TRIP": Chem.BondType.TRIPLE}


def read_component(component_path: Path) -> Chem.Mol:
    """Sanitized RDKit mol with explicit Hs, CIF atom order, atom properties and the ideal coordinates as chirality source."""
    block = gemmi.cif.read(str(component_path))[0]
    columns = ["atom_id", "alt_atom_id", "type_symbol", "charge", "pdbx_leaving_atom_flag"]
    ideal_columns = [f"pdbx_model_Cartn_{axis}_ideal" for axis in "xyz"]
    editable = Chem.RWMol()
    ideal = []
    for row in block.find("_chem_comp_atom.", columns + ideal_columns):
        atom = Chem.Atom(row[2].capitalize())
        atom.SetFormalCharge(int(row[3]))
        atom.SetProp("name", gemmi.cif.as_string(row[0]))
        atom.SetProp("alt_name", gemmi.cif.as_string(row[1]))
        atom.SetProp("leaving_atom", "1" if row[4] == "Y" else "0")
        editable.AddAtom(atom)
        ideal.append([float(row[position]) for position in range(len(columns), len(columns) + 3)])
    index = {atom.GetProp("name"): atom.GetIdx() for atom in editable.GetAtoms()}
    for row in block.find("_chem_comp_bond.", ["atom_id_1", "atom_id_2", "value_order"]):
        editable.AddBond(index[gemmi.cif.as_string(row[0])], index[gemmi.cif.as_string(row[1])], BOND_ORDERS[row[2]])
    mol = editable.GetMol()
    Chem.SanitizeMol(mol)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for atom_index, position in enumerate(ideal):
        conformer.SetAtomPosition(atom_index, position)
    conformer.Set3D(True)
    mol.AddConformer(conformer)
    Chem.AssignStereochemistryFrom3D(mol)
    return mol


def automorphisms(mol: Chem.Mol) -> tuple[tuple[int, ...], ...]:
    """Chirality-preserving atom permutations (identity first), as in the cached `symmetries` property."""
    matches = mol.GetSubstructMatches(mol, uniquify=False, useChirality=True, maxMatches=MAX_SYMMETRIES)
    identity = tuple(range(mol.GetNumAtoms()))
    return (identity, *sorted(match for match in matches if match != identity))


def constraint_properties(mol: Chem.Mol) -> dict[str, np.ndarray]:
    """Cached constraint arrays, from Boltz's parse_ccd_residue helpers with an identity atom map."""
    idx_map = {index: index for index in range(mol.GetNumAtoms())}
    bounds = compute_geometry_constraints(mol, idx_map)
    chiral = compute_chiral_atom_constraints(mol, idx_map)
    stereo = compute_stereo_bond_constraints(mol, idx_map)
    planar_bonds, rings_5, rings_6 = compute_flatness_constraints(mol, idx_map)

    def index_array(items: list, width: int) -> np.ndarray:
        return np.array([item.atom_idxs for item in items], dtype=np.int64).reshape(-1, width).T

    return {
        "pb_edge_index": index_array(bounds, 2),
        "pb_lower_bounds": np.array([item.lower_bound for item in bounds], dtype=np.float64),
        "pb_upper_bounds": np.array([item.upper_bound for item in bounds], dtype=np.float64),
        "pb_bond_mask": np.array([item.is_bond for item in bounds], dtype=bool),
        "pb_angle_mask": np.array([item.is_angle for item in bounds], dtype=bool),
        "chiral_atom_index": index_array(chiral, 4),
        "chiral_check_mask": np.array([item.is_reference for item in chiral], dtype=bool),
        "chiral_atom_orientations": np.array([item.is_r for item in chiral], dtype=bool),
        "stereo_bond_index": index_array(stereo, 4),
        "stereo_check_mask": np.array([item.is_check for item in stereo], dtype=bool),
        "stereo_bond_orientations": np.array([item.is_e for item in stereo], dtype=bool),
        "aromatic_5_ring_index": index_array(rings_5, 5),
        "aromatic_6_ring_index": index_array(rings_6, 6),
        "planar_double_bond_index": index_array(planar_bonds, 6),
    }


def build_ccd_mol(code: str, component_path: Path, seed: int = 0) -> Chem.Mol:
    """Heavy-atom molecule for `code` in the Boltz CCD cache format."""
    mol = Chem.RemoveHs(read_component(component_path))
    mol.RemoveAllConformers()
    mol.SetProp("MOL_NAME", code)
    options = AllChem.ETKDGv3()
    options.randomSeed = seed
    if AllChem.EmbedMultipleConfs(mol, NUM_CONFORMERS, options) == []:
        options.useRandomCoords = True
        AllChem.EmbedMultipleConfs(mol, NUM_CONFORMERS, options)
    assert mol.GetNumConformers() > 0, f"no conformers for {code}"
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    mol.SetProp("symmetries", pickle.dumps(automorphisms(mol)).hex())
    for name, value in constraint_properties(mol).items():
        mol.SetProp(name, pickle.dumps(value).hex())
    return mol

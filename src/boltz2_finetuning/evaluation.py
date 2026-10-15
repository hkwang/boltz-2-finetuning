"""Held-out metrics for Mac1 co-folding predictions: ligand pose RMSD against the deposited structure, and affinity vs IC50."""

import json
import pickle
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from boltz.data.parse.schema import standardize
from rdkit import Chem
from scipy import stats

POCKET_CUTOFF = 6.0
MAX_MATCHES = 1000
BOOTSTRAPS = 2000


def kabsch(moving: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation that map `moving` onto `fixed` (both N x 3) with least squared error: moved = x @ rotation.T + translation."""
    moving_center = moving.mean(axis=0)
    fixed_center = fixed.mean(axis=0)
    left, _, right_t = np.linalg.svd((moving - moving_center).T @ (fixed - fixed_center))
    handedness = np.sign(np.linalg.det(right_t.T @ left.T))
    rotation = right_t.T @ np.diag([1.0, 1.0, handedness]) @ left.T
    return rotation, fixed_center - moving_center @ rotation.T


def rmsd(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(((first - second) ** 2).sum(axis=-1).mean()))


def read_structure(path: Path) -> gemmi.Structure:
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    return structure


def first_altloc(atom: gemmi.Atom) -> bool:
    return atom.altloc in ("\0", "A")


def position(atom: gemmi.Atom) -> list[float]:
    return [atom.pos.x, atom.pos.y, atom.pos.z]


def polymer_residues(chain: gemmi.Chain) -> dict[int, tuple[np.ndarray | None, np.ndarray]]:
    """label_seq -> (CA position or None, heavy-atom positions) for the polymer residues of a chain."""
    residues = {}
    for residue in chain.get_polymer():
        atoms = [atom for atom in residue if first_altloc(atom) and atom.element.name != "H"]
        alpha_carbon = next((np.array(position(atom)) for atom in atoms if atom.name == "CA"), None)
        residues[residue.label_seq] = (alpha_carbon, np.array([position(atom) for atom in atoms]))
    return residues


def ligand_copies(structure: gemmi.Structure, code: str, atom_names: list[str]) -> list[tuple[np.ndarray, float]]:
    """(coordinates in `atom_names` order, mean occupancy) of every fully modelled conformer of ligand `code`.

    Hydrogens are ignored. A residue with alternate locations yields one conformer per altloc label (atoms without an
    altloc plus that label's atoms), so conformers are never mixed; each conformer is scored separately.
    """
    copies = []
    for chain in structure[0]:
        for residue in chain:
            if residue.name != code:
                continue
            heavy = [atom for atom in residue if atom.element.name not in ("H", "D")]
            labels = sorted({atom.altloc for atom in heavy if atom.altloc != "\0"}) or ["\0"]
            for label in labels:
                chosen = {atom.name: atom for atom in heavy if atom.altloc in ("\0", label)}
                if all(name in chosen for name in atom_names):
                    coordinates = np.array([position(chosen[name]) for name in atom_names])
                    copies.append((coordinates, float(np.mean([chosen[name].occ for name in atom_names]))))
    return copies


def load_component(record_id: str, code: str, inputs_dir: Path, ccd_dir: Path) -> Chem.Mol:
    """Heavy-atom Boltz-format ligand mol: the record's extra mols (built A1xxx codes) first, then Boltz's CCD cache."""
    with (inputs_dir / "mols" / f"{record_id}.pkl").open("rb") as handle:
        extras = pickle.load(handle)
    if code in extras:
        return Chem.RemoveHs(extras[code])
    with (ccd_dir / f"{code}.pkl").open("rb") as handle:
        return Chem.RemoveHs(pickle.load(handle))


def pocket_of(proteins: dict[str, dict], ligand: np.ndarray) -> tuple[str, list[int]]:
    """Chain with the most residues within POCKET_CUTOFF of the ligand, and those residues' label_seq."""
    pockets = {}
    for chain_name, residues in proteins.items():
        pockets[chain_name] = [
            seq
            for seq, (_, atoms) in residues.items()
            if np.linalg.norm(atoms[:, None] - ligand[None], axis=-1).min() < POCKET_CUTOFF
        ]
    best = max(pockets, key=lambda chain_name: len(pockets[chain_name]))
    return best, pockets[best]


def common_alpha_carbons(predicted: dict, deposited: dict, sequence_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    shared = [seq for seq in sequence_ids if seq in predicted and predicted[seq][0] is not None and deposited[seq][0] is not None]
    return np.array([predicted[seq][0] for seq in shared]), np.array([deposited[seq][0] for seq in shared])


def pose_metrics(predicted_path: Path, deposited_path: Path, code: str, component: Chem.Mol, smiles: str) -> dict:
    """Pocket-aligned, symmetry-aware heavy-atom ligand RMSD of a Boltz prediction against the deposited ligand.

    `component` is the Boltz-format ligand mol (heavy atoms, `name` property) used to read deposited coordinates, and `smiles`
    is the SMILES that was given to Boltz; Boltz standardizes it before parsing (`parse_boltz_schema` does this for affinity ligands), so predicted ligand atoms follow the atom order of `standardize(smiles)`. The prediction is superposed on the
    deposited protein chain by the C-alpha atoms of residues within POCKET_CUTOFF of the deposited ligand, then the RMSD is the
    minimum over atom-symmetry mappings. With several deposited copies or altloc conformers the closest is reported, with its mean occupancy; `n_copies` counts them all.
    """
    query = Chem.MolFromSmiles(standardize(smiles))
    matches = component.GetSubstructMatches(query, uniquify=False, useChirality=True, maxMatches=MAX_MATCHES)
    if not matches:
        raise ValueError(f"SMILES does not match component {code}")
    names = [atom.GetProp("name") for atom in component.GetAtoms()]
    predicted = read_structure(predicted_path)
    predicted_atoms = [atom for chain in predicted[0] if not chain.get_polymer() for residue in chain for atom in residue]
    predicted_ligand = np.array([position(atom) for atom in predicted_atoms])
    expected_elements = [atom.GetSymbol().upper() for atom in query.GetAtoms()]
    if [atom.element.name.upper() for atom in predicted_atoms] != expected_elements:
        raise ValueError("predicted ligand atoms do not follow the SMILES atom order")
    predicted_protein = next(polymer_residues(chain) for chain in predicted[0] if chain.get_polymer())
    deposited = read_structure(deposited_path)
    deposited_proteins = {chain.name: polymer_residues(chain) for chain in deposited[0] if chain.get_polymer()}
    results = []
    for copy, occupancy in ligand_copies(deposited, code, names):
        chain_name, pocket = pocket_of(deposited_proteins, copy)
        moving, fixed = common_alpha_carbons(predicted_protein, deposited_proteins[chain_name], pocket)
        rotation, translation = kabsch(moving, fixed)
        moved_ligand = predicted_ligand @ rotation.T + translation
        ligand_rmsd = min(rmsd(moved_ligand, copy[list(match)]) for match in matches)
        moving_all, fixed_all = common_alpha_carbons(predicted_protein, deposited_proteins[chain_name], list(deposited_proteins[chain_name]))
        rotation_all, translation_all = kabsch(moving_all, fixed_all)
        results.append(
            {
                "ligand_rmsd": ligand_rmsd,
                "pocket_ca_rmsd": rmsd(moving @ rotation.T + translation, fixed),
                "ca_rmsd": rmsd(moving_all @ rotation_all.T + translation_all, fixed_all),
                "n_pocket_residues": len(moving),
                "deposited_occupancy": occupancy,
            }
        )
    if not results:
        raise ValueError(f"no fully modelled deposited copy of {code}")
    best = min(results, key=lambda result: result["ligand_rmsd"])
    return {**best, "n_copies": len(results), "n_symmetry_matches": len(matches)}


def read_prediction(prediction_dir: Path, name: str) -> list[dict]:
    """Per-sample confidence for each structure sample of one complex, best confidence_score first."""
    samples = []
    for confidence_path in prediction_dir.glob(f"confidence_{name}_model_*.json"):
        model_index = int(confidence_path.stem.rsplit("_", 1)[1])
        samples.append({"model": model_index, **json.loads(confidence_path.read_text())})
    return sorted(samples, key=lambda sample: -sample["confidence_score"])


def bootstrap_interval(values: np.ndarray, statistic, seed: int = 0) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    draws = [statistic(values[generator.integers(0, len(values), len(values))]) for _ in range(BOOTSTRAPS)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def pose_summary(table: pd.DataFrame) -> dict:
    summary = {"n": int(len(table))}
    for column in ("rmsd_top1", "rmsd_best"):
        values = table[column].dropna().to_numpy()
        low, high = bootstrap_interval(values, np.median)
        summary[column] = {
            "median": float(np.median(values)),
            "median_95ci": [low, high],
            "mean": float(values.mean()),
            "frac_below_2A": float((values < 2.0).mean()),
            "frac_below_1A": float((values < 1.0).mean()),
            "n": int(len(values)),
        }
    return summary


def affinity_summary(table: pd.DataFrame) -> dict:
    """Boltz `affinity_pred_value` (log10 IC50 in uM) vs experiment; correlations on measured (=) IC50s only.

    Censored (>) rows are scored one-sidedly: a prediction at or above log10(ceiling) is consistent with the label.
    """
    measured = table[table["ic50_qualifier"] == "="].dropna(subset=["affinity_pred_value"])
    predicted = measured["affinity_pred_value"].to_numpy()
    observed = measured["log10_ic50_uM"].to_numpy()
    indices = np.arange(len(measured))
    spearman = lambda picks: stats.spearmanr(predicted[picks.astype(int)], observed[picks.astype(int)]).statistic
    pearson = lambda picks: stats.pearsonr(predicted[picks.astype(int)], observed[picks.astype(int)]).statistic
    censored = table[table["ic50_qualifier"] == ">"].dropna(subset=["affinity_pred_value"])
    shortfall = np.clip(censored["log10_ic50_uM"] - censored["affinity_pred_value"], 0, None)
    return {
        "n_measured": int(len(measured)),
        "spearman": float(spearman(indices)),
        "spearman_95ci": list(bootstrap_interval(indices, spearman)),
        "pearson": float(pearson(indices)),
        "pearson_95ci": list(bootstrap_interval(indices, pearson)),
        "rmse_log10": float(np.sqrt(((predicted - observed) ** 2).mean())),
        "mae_log10": float(np.abs(predicted - observed).mean()),
        "mean_bias_log10": float((predicted - observed).mean()),
        "n_censored": int(len(censored)),
        "censored_consistent": int((shortfall == 0).sum()),
        "censored_mean_shortfall_log10": float(shortfall.mean()) if len(censored) else None,
    }

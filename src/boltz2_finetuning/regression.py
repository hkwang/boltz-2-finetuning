"""Score Boltz-2 co-folds of deposited complexes against the crystal structures, and build a Boltz checkpoint from fine-tuned weights."""

from pathlib import Path

import gemmi
import numpy as np
import torch

POCKET_RADIUS_A = 10.0


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and shift that least-squares map `mobile` onto `target` (both N x 3): `mobile @ rotation + shift`."""
    mobile_center, target_center = mobile.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((mobile - mobile_center).T @ (target - target_center))
    sign = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, sign]) @ vt
    return rotation, target_center - mobile_center @ rotation


def rmsd(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(((first - second) ** 2).sum(1).mean()))


def read_structure(path: Path) -> gemmi.Structure:
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    structure.remove_hydrogens()
    return structure


def polymer_sequence(structure: gemmi.Structure, chain_name: str) -> str:
    """One-letter SEQRES sequence of the entity of `chain_name`; residue i matches `label_seq == i + 1`."""
    subchain = structure[0][chain_name][0].subchain
    entity = next(entity for entity in structure.entities if subchain in entity.subchains)
    letters = [gemmi.find_tabulated_residue(name).one_letter_code.upper() for name in entity.full_sequence]
    assert "X" not in letters and " " not in letters, f"non-standard residue in {structure.name} chain {chain_name}"
    return "".join(letters)


def reference_ca(structure: gemmi.Structure, chain_name: str) -> dict[int, np.ndarray]:
    """CA coordinates of the observed residues, keyed by 1-based SEQRES index (`label_seq`)."""
    coords = {}
    for residue in structure[0][chain_name]:
        if residue.het_flag == "A" and residue.label_seq is not None and residue.find_atom("CA", "*"):
            coords[residue.label_seq] = np.array(residue["CA"][0].pos.tolist())
    return coords


def reference_ligand(structure: gemmi.Structure, chain_name: str, comp_id: str) -> dict[str, np.ndarray]:
    """Heavy-atom coordinates of the first `comp_id` residue in `chain_name`, keyed by atom name."""
    residue = next(residue for residue in structure[0][chain_name] if residue.name == comp_id)
    return {atom.name: np.array(atom.pos.tolist()) for atom in residue}


def predicted_ca(structure: gemmi.Structure, chain_name: str) -> dict[int, np.ndarray]:
    """CA coordinates of a Boltz prediction, keyed by 1-based residue number (Boltz numbers residues from 1)."""
    return {
        residue.seqid.num: np.array(residue["CA"][0].pos.tolist())
        for residue in structure[0][chain_name]
        if residue.find_atom("CA", "*")
    }


def predicted_ligand(structure: gemmi.Structure, chain_name: str) -> dict[str, np.ndarray]:
    residue = structure[0][chain_name][0]
    return {atom.name: np.array(atom.pos.tolist()) for atom in residue}


def compare_to_reference(
    reference_ca_coords: dict[int, np.ndarray],
    reference_ligand_coords: dict[str, np.ndarray],
    predicted_ca_coords: dict[int, np.ndarray],
    predicted_ligand_coords: dict[str, np.ndarray],
) -> dict[str, float]:
    """CA RMSD, and ligand heavy-atom RMSD in two frames.

    `ligand_rmsd_global`: after superposing all matched CA atoms. `ligand_rmsd_pocket`: after superposing only the CA atoms
    within POCKET_RADIUS_A of the reference ligand. Ligand atoms are paired by CCD atom name, so atoms that are chemically
    equivalent by symmetry (e.g. a flipped phenyl ring) are not swapped, which can inflate RMSD.
    """
    residues = sorted(set(reference_ca_coords) & set(predicted_ca_coords))
    reference_cas = np.array([reference_ca_coords[i] for i in residues])
    predicted_cas = np.array([predicted_ca_coords[i] for i in residues])
    atoms = sorted(set(reference_ligand_coords) & set(predicted_ligand_coords))
    assert len(atoms) == len(reference_ligand_coords), "ligand atoms missing from the prediction"
    reference_atoms = np.array([reference_ligand_coords[name] for name in atoms])
    predicted_atoms = np.array([predicted_ligand_coords[name] for name in atoms])

    rotation, shift = kabsch(predicted_cas, reference_cas)
    near_ligand = np.linalg.norm(reference_cas[:, None] - reference_atoms[None], axis=2).min(1) <= POCKET_RADIUS_A
    pocket_rotation, pocket_shift = kabsch(predicted_cas[near_ligand], reference_cas[near_ligand])
    return {
        "n_residues": len(residues),
        "n_pocket_residues": int(near_ligand.sum()),
        "ca_rmsd": rmsd(predicted_cas @ rotation + shift, reference_cas),
        "ligand_rmsd_global": rmsd(predicted_atoms @ rotation + shift, reference_atoms),
        "ligand_rmsd_pocket": rmsd(predicted_atoms @ pocket_rotation + pocket_shift, reference_atoms),
    }


def write_finetuned_checkpoint(base_checkpoint: Path, trainable_weights: Path, output: Path) -> int:
    """Copy `base_checkpoint` with the tensors in `trainable_weights` (saved by train_diffusion.py) swapped in.

    The result is a normal Boltz-2 checkpoint, so `boltz predict --checkpoint` runs it unchanged. Returns the number of
    replaced tensors. Fails if a name or shape does not match, or if no tensor actually changed.
    """
    checkpoint = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    weights = torch.load(trainable_weights, map_location="cpu")
    changed = 0
    for name, tensor in weights.items():
        assert state_dict[name].shape == tensor.shape, f"shape mismatch for {name}"
        changed += int(not torch.equal(state_dict[name], tensor.to(state_dict[name].dtype)))
        state_dict[name] = tensor.to(state_dict[name].dtype)
    assert changed > 0, "trainable weights are identical to the base checkpoint"
    torch.save(checkpoint, output)
    return len(weights)


def perturbed_weights(base_checkpoint: Path, modules: list[str], noise_scale: float, seed: int) -> dict[str, torch.Tensor]:
    """Base-checkpoint tensors of the given top-level modules plus Gaussian noise of `noise_scale` x each tensor's own std.

    Same format as `trainable_weights.pt`. A positive control for the regression check: weights this noisy must be flagged.
    """
    state_dict = torch.load(base_checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    generator = torch.Generator().manual_seed(seed)
    weights = {}
    for name, tensor in state_dict.items():
        if name.split(".")[0] in modules and tensor.is_floating_point():
            noise = torch.randn(tensor.shape, generator=generator, dtype=torch.float32)
            weights[name] = (tensor.float() + noise_scale * tensor.float().std().nan_to_num() * noise).to(tensor.dtype)
    return weights

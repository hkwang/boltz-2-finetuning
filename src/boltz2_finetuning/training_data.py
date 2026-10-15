"""Boltz-2 diffusion fine-tuning data: crystal records from `data/processed/boltz_inputs/`.

Adapted from the Boltz2_affinity fork's `trainingv3.py` (itself built on `inferencev2.load_input`), keeping the parts
a structure-only run needs: tokenize -> crop -> `Boltz2Featurizer.process(training=True)` -> collate.
Differences from the fork: structures come from `structures/{id}.npz` (deposited coordinates, not re-predicted poses),
extra ligand molecules from `mols/{id}.pkl`, no YAML or cached embeddings, no affinity features.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from boltz.data import const
from boltz.data.crop.boltz import BoltzCropper
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.module.inferencev2 import collate, load_input
from boltz.data.mol import load_canonicals, load_molecules
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.types import Manifest
from rdkit.Chem import Descriptors  # noqa: F401
from torch.utils.data import DataLoader, Dataset


def diffusion_record_ids(records_csv: Path) -> list[str]:
    """Crystal records flagged `diffusion_train` (val/test affinity entries are excluded there)."""
    table = pd.read_csv(records_csv)
    return sorted(table.loc[table["diffusion_train"] & (table["kind"] == "crystal"), "record_id"])


class CrystalDataset(Dataset):
    """Crystal structures as Boltz-2 training features. Random crops, pocket conditioning and MSA subsampling
    draw from a generator seeded from torch's RNG, so `seed_everything(seed, workers=True)` makes epochs reproducible."""

    def __init__(self, inputs_dir: Path, mol_dir: Path, record_ids: list[str], max_tokens: int, max_atoms: int,
                 max_seqs: int, pocket_prob: float, pocket_geometric_p: float, pad_to_max: bool) -> None:
        manifest = Manifest.load(inputs_dir / "manifest.json")
        by_id = {record.id: record for record in manifest.records}
        self.records = [by_id[record_id] for record_id in record_ids]
        self.inputs_dir = inputs_dir
        self.mol_dir = mol_dir
        self.canonicals = load_canonicals(mol_dir)
        self.tokenizer = Boltz2Tokenizer()
        self.featurizer = Boltz2Featurizer()
        self.cropper = BoltzCropper()
        self.max_tokens = max_tokens
        self.max_atoms = max_atoms
        self.max_seqs = max_seqs
        self.pocket_prob = pocket_prob
        self.pocket_geometric_p = pocket_geometric_p
        self.pad_to_max = pad_to_max

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        seed = int(torch.randint(2**31 - 1, (1,)))
        input_data = load_input(
            record,
            self.inputs_dir / "structures",
            self.inputs_dir / "msa",
            extra_mols_dir=self.inputs_dir / "mols",
        )
        tokenized = self.tokenizer.tokenize(input_data)
        tokenized = self.cropper.crop(
            tokenized, max_tokens=self.max_tokens, random=np.random.RandomState(seed), max_atoms=self.max_atoms
        )
        molecules = {**self.canonicals, **input_data.extra_mols}
        missing = set(tokenized.tokens["res_name"].tolist()) - set(molecules)
        molecules.update(load_molecules(self.mol_dir, list(missing)))
        features = self.featurizer.process(
            tokenized,
            molecules=molecules,
            random=np.random.default_rng(seed),
            training=True,
            max_tokens=self.max_tokens if self.pad_to_max else None,
            max_atoms=self.max_atoms if self.pad_to_max else None,
            max_seqs=self.max_seqs,
            pad_to_max_seqs=False,
            binder_pocket_conditioned_prop=self.pocket_prob,
            binder_pocket_sampling_geometric_p=self.pocket_geometric_p,
            compute_frames=True,
        )
        features["record_id"] = record.id
        return features


def collate_with_ids(items: list[dict]) -> dict:
    """Boltz's collate plus a list of record ids; Lightning cannot move a `Record` dataclass to the GPU."""
    record_ids = [item.pop("record_id") for item in items]
    batch = collate(items)
    batch["record_id"] = record_ids
    return batch


class CrystalDataModule(pl.LightningDataModule):
    """Training-only data module (batch size 1: the confidence-path symmetry code in `Boltz2.training_step` requires it)."""

    def __init__(self, dataset: CrystalDataset, num_workers: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset, batch_size=1, shuffle=True, num_workers=self.num_workers, collate_fn=collate_with_ids,
            persistent_workers=self.num_workers > 0,
        )

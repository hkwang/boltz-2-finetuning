"""Affinity-head fine-tuning on cached Boltz-2 trunk features.

`boltz predict` feeds each `AffinityModule` three things: the trunk pair representation `z` (masked to ligand-protein and
ligand-ligand pairs), the input embedding `s_inputs`, and the representative-atom coordinates of a predicted complex. With the
trunk and structure module frozen these are fixed per complex, so `experiments/cache_affinity_features.py` computes them once
and this module trains the two ensemble `AffinityModule`s on the cache with large, assay-grouped batches (stock Boltz has no
affinity loss; the loss follows `molecularinformatics/Boltz2_affinity`: Huber on log10 IC50 in uM, pairwise differences within an
assay, focal loss on binder / non-binder).

Cache file per complex (`.npz`): `s_inputs` [N, 384], `z_ligand_rows` [L, N, 128], `z_ligand_cols` [N, L, 128] (the only non-zero
parts of the masked `z`), `rep_coords` [N, 3], `is_ligand` [N], `is_receptor` [N], `mw`.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from boltz.model.modules.affinity import AffinityModule
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset, Sampler


def huber(prediction: Tensor, target: Tensor, delta: float) -> Tensor:
    return functional.huber_loss(prediction, target, reduction="none", delta=delta)


def focal_loss(logits: Tensor, targets: Tensor, alpha: float, gamma: float) -> Tensor:
    """Per-sample focal loss; `alpha` weights the positive class."""
    cross_entropy = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability_true = torch.exp(-cross_entropy)
    class_weight = alpha * targets + (1 - alpha) * (1 - targets)
    return class_weight * (1 - probability_true) ** gamma * cross_entropy


def pairwise_loss(prediction: Tensor, target: Tensor, weight: Tensor, assay: list[str], delta: float) -> Tensor:
    """Huber loss on predicted vs measured differences for all pairs with the same assay id; 0 with no pair."""
    index_a, index_b = [], []
    for first in range(len(assay)):
        for second in range(first + 1, len(assay)):
            if assay[first] == assay[second]:
                index_a.append(first)
                index_b.append(second)
    if not index_a:
        return prediction.sum() * 0
    index_a, index_b = torch.tensor(index_a, device=prediction.device), torch.tensor(index_b, device=prediction.device)
    pair_weight = (weight[index_a] + weight[index_b]) / 2
    losses = huber(prediction[index_a] - prediction[index_b], target[index_a] - target[index_b], delta)
    return (losses * pair_weight).sum() / pair_weight.sum()


class AffinityCacheDataset(Dataset):
    """Cached features plus labels from `labels.csv` (experiments/prepare_affinity_inputs.py) for the given record ids."""

    def __init__(self, cache_dir: Path, labels: pd.DataFrame) -> None:
        self.cache_dir = cache_dir
        self.labels = labels.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        row = self.labels.iloc[index]
        with np.load(self.cache_dir / f"{row.record_id}.npz") as cached:
            features = {name: cached[name] for name in cached.files}
        features.update(
            record_id=row.record_id, assay_id=row.assay_id, kind=row.kind, weight=float(row.weight), binary=float(row.binary),
            log10_ic50_uM=float(row.log10_ic50_uM) if row.kind == "ic50" and not row.censored else float("nan"),
        )
        return features


def collate_affinity(items: list[dict]) -> dict:
    """Pad to the largest complex. Dense `z` is rebuilt from the cached ligand rows and columns."""
    size = max(item["s_inputs"].shape[0] for item in items)
    batch = len(items)
    s_inputs = torch.zeros(batch, size, items[0]["s_inputs"].shape[1])
    z = torch.zeros(batch, size, size, items[0]["z_ligand_rows"].shape[2])
    rep_coords = torch.zeros(batch, size, 3)
    is_ligand = torch.zeros(batch, size, dtype=torch.bool)
    is_receptor = torch.zeros(batch, size, dtype=torch.bool)
    pad_mask = torch.zeros(batch, size, dtype=torch.bool)
    for position, item in enumerate(items):
        tokens = item["s_inputs"].shape[0]
        ligand = torch.from_numpy(item["is_ligand"])
        s_inputs[position, :tokens] = torch.from_numpy(item["s_inputs"])
        pair = z[position, :tokens, :tokens]
        pair[ligand] = torch.from_numpy(item["z_ligand_rows"]).float()
        pair[:, ligand] = torch.from_numpy(item["z_ligand_cols"]).float()
        rep_coords[position, :tokens] = torch.from_numpy(item["rep_coords"])
        is_ligand[position, :tokens] = ligand
        is_receptor[position, :tokens] = torch.from_numpy(item["is_receptor"])
        pad_mask[position, :tokens] = True
    return {
        "s_inputs": s_inputs, "z": z, "rep_coords": rep_coords, "is_ligand": is_ligand, "is_receptor": is_receptor, "pad_mask": pad_mask,
        "weight": torch.tensor([item["weight"] for item in items]), "binary": torch.tensor([item["binary"] for item in items]),
        "log10_ic50_uM": torch.tensor([item["log10_ic50_uM"] for item in items]),
        "assay_id": [item["assay_id"] for item in items], "kind": [item["kind"] for item in items],
        "record_id": [item["record_id"] for item in items],
    }


class MixedBatchSampler(Sampler):
    """Batches of `n_ic50` IC50 rows from one assay (so pairs exist) plus `n_hit` binder and `n_nonhit` non-binder rows.

    Sampled with replacement; each rank draws its own stream (seed + rank), `steps` batches per epoch.
    """

    def __init__(self, kinds: list[str], assays: list[str], n_ic50: int, n_hit: int, n_nonhit: int, steps: int, seed: int, rank: int) -> None:
        self.by_kind = {kind: [i for i, value in enumerate(kinds) if value == kind] for kind in ("ic50", "hit", "nonhit")}
        self.ic50_by_assay: dict[str, list[int]] = {}
        for index in self.by_kind["ic50"]:
            self.ic50_by_assay.setdefault(assays[index], []).append(index)
        self.counts = {"ic50": n_ic50, "hit": n_hit, "nonhit": n_nonhit}
        self.steps, self.generator = steps, random.Random(seed + 1000 * rank)

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        assay_names = list(self.ic50_by_assay)
        assay_weights = [len(self.ic50_by_assay[name]) for name in assay_names]
        for _ in range(self.steps):
            assay = self.generator.choices(assay_names, assay_weights)[0]
            pool = self.ic50_by_assay[assay]
            batch = self.generator.choices(pool, k=self.counts["ic50"]) if len(pool) < self.counts["ic50"] else self.generator.sample(pool, self.counts["ic50"])
            batch += self.generator.choices(self.by_kind["hit"], k=self.counts["hit"])
            batch += self.generator.choices(self.by_kind["nonhit"], k=self.counts["nonhit"])
            yield batch


class AffinityHead(pl.LightningModule):
    """The two ensemble `AffinityModule`s of Boltz-2, trained with the weighted Huber + pairwise + focal loss."""

    def __init__(self, hyper_parameters: dict, loss_args: dict, max_lr: float, warmup_steps: int, weight_decay: float) -> None:
        super().__init__()
        self.module1 = AffinityModule(hyper_parameters["token_s"], hyper_parameters["token_z"], **hyper_parameters["affinity_model_args1"])
        self.module2 = AffinityModule(hyper_parameters["token_s"], hyper_parameters["token_z"], **hyper_parameters["affinity_model_args2"])
        self.loss_args, self.max_lr, self.warmup_steps, self.weight_decay = loss_args, max_lr, warmup_steps, weight_decay
        self.validation_rows: list[dict] = []

    def on_train_start(self) -> None:
        """The cuEquivariance triangle-attention backward refuses FP32 unless TF32 is on; set it on every rank right before the first step."""
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def load_pretrained(self, state_dict: dict[str, Tensor]) -> None:
        for name, module in (("affinity_module1", self.module1), ("affinity_module2", self.module2)):
            module.load_state_dict({key[len(name) + 1:]: value for key, value in state_dict.items() if key.startswith(name + ".")}, strict=True)

    def predict_batch(self, batch: dict) -> dict[str, Tensor]:
        """Ensemble mean of value (log10 IC50 uM) and binary logit, as `Boltz2.forward` averages value and probability."""
        batch_size, size = batch["pad_mask"].shape
        feats = {
            "token_to_rep_atom": torch.eye(size, device=self.device).expand(batch_size, size, size),
            "token_pad_mask": batch["pad_mask"].float(),
            "mol_type": torch.where(batch["is_receptor"], 0, 3),
            "affinity_token_mask": batch["is_ligand"].float(),
        }
        outputs = [module(s_inputs=batch["s_inputs"], z=batch["z"], x_pred=batch["rep_coords"], feats=feats, multiplicity=1, use_kernels=True)
                   for module in (self.module1, self.module2)]
        value = torch.stack([out["affinity_pred_value"][:, 0] for out in outputs]).mean(0)
        probability = torch.stack([torch.sigmoid(out["affinity_logits_binary"][:, 0]) for out in outputs]).mean(0)
        logit = torch.logit(probability.clamp(1e-6, 1 - 1e-6))
        return {"value": value, "logit": logit, "probability": probability, "per_module_value": [out["affinity_pred_value"][:, 0] for out in outputs],
                "per_module_logit": [out["affinity_logits_binary"][:, 0] for out in outputs]}

    def loss(self, batch: dict, out: dict) -> tuple[Tensor, dict[str, Tensor]]:
        args = self.loss_args
        total, terms = 0.0, {}
        measured = ~torch.isnan(batch["log10_ic50_uM"])
        if measured.any():
            target, weight = batch["log10_ic50_uM"][measured], batch["weight"][measured]
            assay = [name for name, keep in zip(batch["assay_id"], measured.tolist()) if keep]
            absolute = torch.zeros((), device=self.device)
            pairwise = torch.zeros((), device=self.device)
            for value in out["per_module_value"]:
                prediction = value[measured]
                absolute = absolute + (huber(prediction, target, args["huber_delta"]) * weight).sum() / weight.sum() / 2
                pairwise = pairwise + pairwise_loss(prediction, target, weight, assay, args["huber_delta"]) / 2
            terms["absolute"], terms["pairwise"] = absolute, pairwise
            total = total + args["absolute_weight"] * absolute + args["pairwise_weight"] * pairwise
        binary = torch.zeros((), device=self.device)
        for logits in out["per_module_logit"]:
            binary = binary + (focal_loss(logits, batch["binary"], args["focal_alpha"], args["focal_gamma"]) * batch["weight"]).sum() / batch["weight"].sum() / 2
        terms["binary"] = binary
        return total + args["binary_weight"] * binary, terms

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        loss, terms = self.loss(batch, self.predict_batch(batch))
        self.log("train/loss", loss, batch_size=len(batch["kind"]))
        for name, value in terms.items():
            self.log(f"train/{name}", value, batch_size=len(batch["kind"]))
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        out = self.predict_batch(batch)
        for position, record_id in enumerate(batch["record_id"]):
            self.validation_rows.append({"record_id": record_id, "pred": float(out["value"][position]), "probability": float(out["probability"][position]),
                                         "target": float(batch["log10_ic50_uM"][position]), "binary": float(batch["binary"][position])})

    def on_validation_epoch_end(self) -> None:
        table = pd.DataFrame(self.validation_rows)
        self.validation_rows = []
        measured = table.dropna(subset=["target"])
        if len(measured) > 2:
            self.log("val/spearman", float(measured["pred"].corr(measured["target"], method="spearman")), rank_zero_only=True)
            self.log("val/rmse", float(np.sqrt(((measured["pred"] - measured["target"]) ** 2).mean())), rank_zero_only=True)
            self.log("val/bias", float((measured["pred"] - measured["target"]).mean()), rank_zero_only=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.max_lr, weight_decay=self.weight_decay, betas=(0.9, 0.95))
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / self.warmup_steps))
        return [optimizer], [{"scheduler": schedule, "interval": "step"}]

    def exported_weights(self) -> dict[str, Tensor]:
        """State dict under the keys of `boltz2_aff.ckpt`."""
        return {**{f"affinity_module1.{k}": v.detach().cpu() for k, v in self.module1.state_dict().items()},
                **{f"affinity_module2.{k}": v.detach().cpu() for k, v in self.module2.state_dict().items()}}


def make_loader(dataset: AffinityCacheDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_affinity)

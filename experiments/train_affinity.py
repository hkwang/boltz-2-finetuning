"""Fine-tune Boltz-2's two affinity modules on cached trunk features (see src/boltz2_finetuning/affinity.py).

Run: PYTHONPATH=src python experiments/train_affinity.py configs/finetune_affinity.yaml [key.sub=value ...]
Needs the caches written by experiments/cache_affinity_features.py (training complexes and the 45 held-out entries). Every
`eval_every_n_steps` it scores the held-out entries and writes weights/step<N>.pt (keys of boltz2_aff.ckpt; patch a Boltz checkpoint
with experiments/make_finetuned_checkpoint.py) and eval_step<N>.csv. Only `affinity_module1/2` change; the rest of the checkpoint is untouched.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from boltz2_finetuning.affinity import AffinityCacheDataset, AffinityHead, MixedBatchSampler, collate_affinity
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader


def load_config(argv: list[str]) -> dict:
    config = OmegaConf.create(yaml.safe_load(Path(argv[1]).read_text()))
    return OmegaConf.to_container(OmegaConf.merge(config, OmegaConf.from_dotlist(argv[2:])), resolve=True)


def eval_labels(entries_csv: Path, cache_dir: Path) -> pd.DataFrame:
    """Held-out entries as label rows; censored `>` rows keep a binary label of 0 and no regression target."""
    entries = pd.read_csv(entries_csv)
    return pd.DataFrame({
        "record_id": "mac1_eval_" + entries["pdb_id"], "kind": "ic50", "assay_id": "heldout", "weight": 1.0,
        "binary": ((entries["ic50_qualifier"] == "=") & (10 ** entries["log10_ic50_uM"] <= 100)).astype(int),
        "log10_ic50_uM": entries["log10_ic50_uM"], "censored": entries["ic50_qualifier"] == ">", "split": entries["split"],
    }).loc[lambda table: [(cache_dir / f"{name}.npz").exists() for name in table["record_id"]]]


class HeldOutEvaluation(Callback):
    """Scores the held-out entries on rank 0 every `every` steps; writes weights and a per-entry CSV."""

    def __init__(self, output_dir: Path, labels: pd.DataFrame, cache_dir: Path, every: int) -> None:
        self.output_dir, self.labels, self.cache_dir, self.every = output_dir, labels, cache_dir, every

    def evaluate(self, module: AffinityHead, step: int) -> None:
        module.eval()
        dataset = AffinityCacheDataset(self.cache_dir, self.labels)
        rows = []
        with torch.no_grad():
            for index in range(len(dataset)):
                batch = collate_affinity([dataset[index]])
                batch = {key: value.to(module.device) if torch.is_tensor(value) else value for key, value in batch.items()}
                out = module.predict_batch(batch)
                rows.append({"record_id": self.labels.record_id.iloc[index], "pred": float(out["value"]), "probability": float(out["probability"])})
        table = self.labels.merge(pd.DataFrame(rows), on="record_id")
        table.to_csv(self.output_dir / f"eval_step{step}.csv", index=False)
        measured = table[~table["censored"]]
        spearman = measured["pred"].corr(measured["log10_ic50_uM"], method="spearman")
        rmse = float(np.sqrt(((measured["pred"] - measured["log10_ic50_uM"]) ** 2).mean()))
        bias = float((measured["pred"] - measured["log10_ic50_uM"]).mean())
        print(f"EVAL step {step}: n {len(measured)} spearman {spearman:.3f} rmse {rmse:.3f} bias {bias:+.3f}", flush=True)
        module.train()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx) -> None:
        step = trainer.global_step
        if trainer.is_global_zero and step % self.every == 0:
            (self.output_dir / "weights").mkdir(exist_ok=True)
            torch.save(module.exported_weights(), self.output_dir / "weights" / f"step{step}.pt")
            self.evaluate(module, step)
        trainer.strategy.barrier()


class StepLog(Callback):
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx) -> None:
        if trainer.is_global_zero and trainer.global_step % 10 == 0:
            metrics = {key.split("/")[1]: float(value) for key, value in trainer.callback_metrics.items() if key.startswith("train/")}
            print(f"step {trainer.global_step} " + " ".join(f"{name} {value:.3f}" for name, value in metrics.items()), flush=True)


def main() -> None:
    config = load_config(sys.argv)
    pl.seed_everything(config["seed"], workers=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    output_dir = Path(config["output_dir"]) / config["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    settings, batch_settings = config["trainer"], config["batch"]

    checkpoint = torch.load(Path(config["checkpoint"]).expanduser(), map_location="cpu", weights_only=False)
    model = AffinityHead(checkpoint["hyper_parameters"], config["loss"], config["optimizer"]["max_lr"], config["optimizer"]["warmup_steps"],
                         config["optimizer"]["weight_decay"])
    model.load_pretrained(checkpoint["state_dict"])
    del checkpoint

    cache_dir = Path(config["cache_dir"])
    labels = pd.read_csv(config["labels_csv"])
    labels = labels[[(cache_dir / f"{name}.npz").exists() for name in labels["record_id"]]].reset_index(drop=True)
    print(labels.groupby("kind").agg(n=("record_id", "size"), binary=("binary", "sum"), censored=("censored", "sum")).to_string(), flush=True)
    dataset = AffinityCacheDataset(cache_dir, labels)
    rank = int(__import__("os").environ.get("LOCAL_RANK", 0))
    sampler = MixedBatchSampler(labels["kind"].tolist(), labels["assay_id"].tolist(), batch_settings["n_ic50"], batch_settings["n_hit"],
                                batch_settings["n_nonhit"], settings["max_steps"], config["seed"], rank)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=batch_settings["num_workers"], collate_fn=collate_affinity)

    eval_cache = Path(config["eval_cache_dir"])
    held_out = eval_labels(Path(config["eval_entries_csv"]), eval_cache)
    callbacks = [StepLog(), HeldOutEvaluation(output_dir, held_out, eval_cache, settings["eval_every_n_steps"])]
    trainer = pl.Trainer(
        accelerator="gpu", devices=settings["devices"], strategy=DDPStrategy(find_unused_parameters=False) if settings["devices"] > 1 else "auto",
        precision="32-true", max_steps=settings["max_steps"], gradient_clip_val=settings["gradient_clip_val"], log_every_n_steps=1,
        num_sanity_val_steps=0, limit_val_batches=0, logger=CSVLogger(str(output_dir), name="logs"), callbacks=callbacks,
        enable_progress_bar=False, enable_checkpointing=False, use_distributed_sampler=False,
    )
    trainer.fit(model, train_dataloaders=loader)
    if trainer.is_global_zero:
        torch.save(model.exported_weights(), output_dir / "trainable_weights.pt")
        print("TRAIN DONE", flush=True)


if __name__ == "__main__":
    main()

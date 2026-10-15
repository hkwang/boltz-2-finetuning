"""Fine-tune the Boltz-2 diffusion head on Mac1 crystal structures.

Run: PYTHONPATH=src python experiments/train_diffusion.py configs/finetune_diffusion.yaml [key.sub=value ...]

Uses the installed `boltz` package unmodified, via `Boltz2Diffusion` (a diffusion-only `training_step`, see `src/boltz2_finetuning/model.py`). Settings the stock class needs for training but the
checkpoint does not carry: `validate_structure=False`, an all-off `steering_args` dict, and `structure_prediction_training=True` (the checkpoint stores False, which makes `training_step` skip the diffusion loss), `confidence_prediction=True`
(otherwise `forward` never builds `diffusion_conditioning` during training). The checkpoint's `mse_rotational_alignment`
diffusion arg and `add_bond_loss` loss arg are dropped because the installed `AtomDiffusion` does not accept them, and `Boltz2.setup` reads `self.validate_structure`, which the class never sets (we set it after loading), and `pairformer_args` gets the `v2: True`
that `boltz predict` also sets (the checkpoint omits it, which builds the wrong pairformer layout). The confidence head is frozen and its loss
weight is 0, so it only costs compute. `Boltz2.training_step` has no affinity loss, so this run is structure-only.
"""

import hashlib
import sys
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from boltz2_finetuning.lora import add_lora, merged_weights
from boltz2_finetuning.model import Boltz2Diffusion
from boltz2_finetuning.training_data import CrystalDataModule, CrystalDataset, diffusion_record_ids
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy

STEERING_OFF = {
    "fk_steering": False,
    "num_particles": 1,
    "fk_lambda": 0.0,
    "fk_resampling_interval": 1,
    "physical_guidance_update": False,
    "contact_guidance_update": False,
    "num_gd_steps": 0,
}


def load_config(argv: list[str]) -> dict:
    config = OmegaConf.create(yaml.safe_load(Path(argv[1]).read_text()))
    return OmegaConf.to_container(OmegaConf.merge(config, OmegaConf.from_dotlist(argv[2:])), resolve=True)


def digest_frozen_parameters(model: torch.nn.Module) -> str:
    """SHA-256 over all parameters with `requires_grad` False, in name order."""
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().float().numpy().tobytes())
    return digest.hexdigest()


def exported_weights(model: torch.nn.Module, uses_lora: bool) -> dict[str, torch.Tensor]:
    """Tensors to write into a Boltz checkpoint, under the original parameter names (LoRA deltas merged into dense weights)."""
    if uses_lora:
        return merged_weights(model)
    return {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}


class SaveWeights(Callback):
    """Writes `weights/step{N}.pt` (input of experiments/make_finetuned_checkpoint.py) every `every` optimizer steps."""

    def __init__(self, directory: Path, every: int, uses_lora: bool) -> None:
        self.directory, self.every, self.uses_lora = directory, every, uses_lora

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx) -> None:
        if trainer.is_global_zero and trainer.global_step % self.every == 0:
            self.directory.mkdir(exist_ok=True)
            torch.save(exported_weights(module, self.uses_lora), self.directory / f"step{trainer.global_step}.pt")


class StepTimer(Callback):
    """Prints seconds per optimizer step and peak GPU memory; skips the first step (kernel JIT warmup)."""

    def on_train_batch_start(self, trainer, module, batch, batch_idx) -> None:
        self.start = time.perf_counter()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx) -> None:
        torch.cuda.synchronize()
        seconds = time.perf_counter() - self.start
        if not outputs:
            raise RuntimeError(f"Boltz2.training_step skipped batch {batch_idx} (see 'Skipping batch' above)")
        loss = float(outputs["loss"])
        memory_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"step {trainer.global_step} record {batch['record_id'][0]} tokens {batch['token_index'].shape[1]} "
              f"atoms {batch['atom_pad_mask'].shape[1]} loss {loss:.4f} {seconds:.2f}s peak {memory_gb:.1f}GB", flush=True)


def main() -> None:
    config = load_config(sys.argv)
    pl.seed_everything(config["seed"], workers=True)
    output_dir = Path(config["output_dir"]) / config["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    hyper_parameters = torch.load(Path(config["checkpoint"]).expanduser(), map_location="cpu", weights_only=False)[
        "hyper_parameters"]
    training_args = OmegaConf.merge(hyper_parameters["training_args"], OmegaConf.create(config["training_args"]))
    diffusion_loss_args = {k: v for k, v in hyper_parameters["diffusion_loss_args"].items() if k != "add_bond_loss"}
    diffusion_process_args = {k: v for k, v in hyper_parameters["diffusion_process_args"].items()
                              if k != "mse_rotational_alignment"}
    model = Boltz2Diffusion.load_from_checkpoint(
        Path(config["checkpoint"]).expanduser(),
        strict=True,
        map_location="cpu",
        training_args=training_args,
        diffusion_process_args=diffusion_process_args,
        diffusion_loss_args=diffusion_loss_args,
        pairformer_args={**hyper_parameters["pairformer_args"], "v2": True},
        validate_structure=False,
        structure_prediction_training=True,
        checkpoint_diffusion_conditioning=config["checkpoint_diffusion_conditioning"],
        score_model_args={**hyper_parameters["score_model_args"],
                          "activation_checkpointing": config["score_activation_checkpointing"]},
        steering_args=STEERING_OFF,
        ema=False,
        use_kernels=True,
    )
    model.validate_structure = False
    lora = config.get("lora")
    trainable = config["trainable_modules"]
    for name, parameter in model.named_parameters():
        parameter.requires_grad = not lora and name.split(".")[0] in trainable
    if lora:
        wrapped = add_lora(model, lora["prefix"], lora["rank"], lora["alpha"])
        print(f"LoRA rank {lora['rank']} alpha {lora['alpha']} on {wrapped} linear layers under {lora['prefix']}")
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {num_trainable / 1e6:.1f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    frozen_before = digest_frozen_parameters(model)

    data = config["data"]
    dataset = CrystalDataset(
        Path(config["inputs_dir"]),
        Path(config["mol_dir"]).expanduser(),
        diffusion_record_ids(Path(config["records_csv"])),
        data["max_tokens"], data["max_atoms"], data["max_seqs"], data["pocket_prob"], data["pocket_geometric_p"],
        data["pad_to_max"],
    )
    print(f"training on {len(dataset)} crystal records")

    settings = config["trainer"]
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=settings["devices"],
        strategy=DDPStrategy(static_graph=settings["ddp_static_graph"]) if settings["devices"] > 1 else "auto",
        precision=settings["precision"],
        max_steps=settings["max_steps"],
        accumulate_grad_batches=settings["accumulate_grad_batches"],
        gradient_clip_val=settings["gradient_clip_val"],
        log_every_n_steps=settings["log_every_n_steps"],
        num_sanity_val_steps=0,
        limit_val_batches=0,
        logger=CSVLogger(str(output_dir), name="logs"),
        callbacks=[
            StepTimer(),
            SaveWeights(output_dir / "weights", settings["save_weights_every_n_steps"], bool(lora)),
            ModelCheckpoint(dirpath=output_dir / "checkpoints", save_last=True, save_top_k=0,
                            every_n_train_steps=settings["checkpoint_every_n_steps"]),
        ],
        enable_progress_bar=False,
    )
    last_checkpoint = output_dir / "checkpoints" / "last.ckpt"
    trainer.fit(model, datamodule=CrystalDataModule(dataset, data["num_workers"]),
                ckpt_path=str(last_checkpoint) if last_checkpoint.exists() else None)

    if trainer.is_global_zero:
        assert frozen_before == digest_frozen_parameters(model), "frozen parameters changed during training"
        print("frozen parameters bit-identical after training")
        torch.save(exported_weights(model, bool(lora)), output_dir / "trainable_weights.pt")


if __name__ == "__main__":
    main()

"""Boltz-2 with a diffusion-only `training_step`; the installed `boltz` package is not modified.

Stock `Boltz2.training_step` cannot train the structure module while the confidence head is on: after `forward` it
calls `get_true_coordinates`, which asserts a batch size of 1 on `batch["coords"]`, but `forward` has by then expanded
`coords` to `diffusion_multiplicity` copies (3-D). `forward` itself needs `confidence_prediction=True` to build the diffusion
conditioning in training. So this subclass keeps `forward` (including the unused confidence pass) and computes only the
distogram and diffusion losses, with no try/except, so a bad batch fails loudly instead of being skipped.
"""

import numpy as np
from boltz.model.loss.distogramv2 import distogram_loss
from boltz.model.models.boltz2 import Boltz2
from torch import Tensor


class Boltz2Diffusion(Boltz2):
    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        args = self.training_args
        recycling_steps = int(np.random.default_rng(self.global_step).integers(0, args.recycling_steps + 1))
        out = self(
            feats=batch,
            recycling_steps=recycling_steps,
            num_sampling_steps=args.sampling_steps,
            multiplicity_diffusion_train=args.diffusion_multiplicity,
            diffusion_samples=args.diffusion_samples,
        )
        disto_loss, _ = distogram_loss(out, batch, aggregate_distogram=self.aggregate_distogram)
        diffusion = self.structure_module.compute_loss(
            batch, out, multiplicity=args.diffusion_multiplicity, **self.diffusion_loss_args
        )
        loss = args.diffusion_loss_weight * diffusion["loss"] + args.distogram_loss_weight * disto_loss
        self.log("train/loss", loss)
        self.log("train/distogram_loss", disto_loss)
        self.log("train/diffusion_loss", diffusion["loss"])
        for name, value in diffusion["loss_breakdown"].items():
            self.log(f"train/{name}", value)
        return loss

    def on_train_epoch_end(self) -> None:
        """Stock hook logs confidence metrics that only exist when the model is built with `validate_structure=True`."""

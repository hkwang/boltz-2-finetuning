"""Cache the frozen-trunk inputs of Boltz-2's affinity modules for complexes whose poses `boltz predict` has already written.

Run: python experiments/cache_affinity_features.py BOLTZ_OUT_DIR CACHE_DIR [--checkpoint boltz2_aff.ckpt] [--recycling-steps 5] [--shard-index I --num-shards K]
BOLTZ_OUT_DIR is a `boltz predict` output directory (processed/ plus predictions/<id>/pre_affinity_<id>.npz, the top-ranked predicted complex).
For each complex this reruns the trunk exactly as the affinity stage of `boltz predict` does (affinity MSA/profile features, the cropper, recycling 5)
and stores what `AffinityModule` reads (see `src/boltz2_finetuning/affinity.py`). Existing cache files are skipped, so the job is resumable.
Per-record pose = the pre_affinity structure (the structure stage's top-ranked sample), not the affinity stage's own resample.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from boltz.data.module.inferencev2 import PredictionDataset, collate
from boltz.data.types import Manifest
from boltz.main import BoltzSteeringParams, Boltz2DiffusionParams, MSAModuleArgs, PairformerArgsV2
from boltz.model.models.boltz2 import Boltz2
from rdkit.Chem import Descriptors  # noqa: F401


def load_trunk_model(checkpoint: Path, device: torch.device) -> Boltz2:
    """Boltz-2 as `boltz predict` builds it for the affinity stage, with the confidence, affinity and structure sampling switched off after loading (only the trunk is needed)."""
    steering = BoltzSteeringParams()
    steering.fk_steering = steering.physical_guidance_update = steering.contact_guidance_update = False
    model = Boltz2.load_from_checkpoint(
        checkpoint, strict=True, map_location="cpu", ema=False, use_kernels=device.type == "cuda",
        predict_args={"recycling_steps": 5, "sampling_steps": 1, "diffusion_samples": 1, "max_parallel_samples": 1},
        diffusion_process_args=asdict(Boltz2DiffusionParams()), pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(MSAModuleArgs(use_paired_feature=True)), steering_args=asdict(steering),
    )
    model.affinity_prediction = False
    model.confidence_prediction = False
    model.skip_run_structure = True
    return model.eval().to(device)


@torch.no_grad()
def cache_record(model: Boltz2, batch: dict, recycling_steps: int) -> dict[str, np.ndarray]:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=batch["token_pad_mask"].is_cuda):
        out = model(batch, recycling_steps=recycling_steps)
        s_inputs = model.input_embedder(batch, affinity=True)
    ligand = batch["affinity_token_mask"][0].bool() & batch["token_pad_mask"][0].bool()
    receptor = (batch["mol_type"][0] == 0) & batch["token_pad_mask"][0].bool()
    pair_mask = ligand[:, None] * receptor[None, :] + receptor[:, None] * ligand[None, :] + ligand[:, None] * ligand[None, :]
    z = out["z"][0] * pair_mask[:, :, None]
    coords = batch["coords"][0, 0].float()
    rep_coords = batch["token_to_rep_atom"][0].float() @ coords
    return {
        "s_inputs": s_inputs[0].float().cpu().numpy(),
        "z_ligand_rows": z[ligand].half().cpu().numpy(),
        "z_ligand_cols": z[:, ligand].half().cpu().numpy(),
        "rep_coords": rep_coords.cpu().numpy(),
        "is_ligand": ligand.cpu().numpy(),
        "is_receptor": receptor.cpu().numpy(),
        "mw": np.float32(batch["affinity_mw"][0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("boltz_out", type=Path)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path.home() / ".cache/boltz/boltz2_aff.ckpt")
    parser.add_argument("--mol-dir", type=Path, default=Path.home() / ".cache/boltz/mols")
    parser.add_argument("--recycling-steps", type=int, default=5)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed = args.boltz_out / "processed"
    manifest = Manifest.load(processed / "manifest.json")
    records = [record for record in manifest.records if (args.boltz_out / "predictions" / record.id / f"pre_affinity_{record.id}.npz").exists()]
    records = sorted(records, key=lambda record: record.id)[args.shard_index::args.num_shards][:args.limit]
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [record for record in records if not (args.cache_dir / f"{record.id}.npz").exists()]
    print(f"{len(records)} complexes with poses in this shard, {len(todo)} to cache", flush=True)

    dataset = PredictionDataset(
        manifest=Manifest(todo), target_dir=args.boltz_out / "predictions", msa_dir=processed / "msa", mol_dir=args.mol_dir,
        constraints_dir=processed / "constraints", template_dir=None, extra_mols_dir=processed / "mols", override_method="other", affinity=True,
    )
    model = load_trunk_model(args.checkpoint, device)
    for index, record in enumerate(todo):
        batch = collate([dataset[index]])
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        cached = cache_record(model, batch, args.recycling_steps)
        np.savez(args.cache_dir / f"{record.id}.npz", **cached)
        if index % 25 == 0:
            print(f"{index + 1}/{len(todo)} {record.id} tokens {cached['s_inputs'].shape[0]} ligand tokens {int(cached['is_ligand'].sum())}", flush=True)
    print("CACHE DONE", flush=True)


if __name__ == "__main__":
    main()

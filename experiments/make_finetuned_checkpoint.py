"""Write Boltz-2 checkpoints whose fine-tuned tensors replace the pretrained ones, for `boltz predict --checkpoint/--affinity_checkpoint`.

Run: python experiments/make_finetuned_checkpoint.py TRAINABLE_WEIGHTS.pt OUT_DIR
`boltz predict` runs the structure module in both stages (structure from `boltz2_conf.ckpt`, then a re-sampled pose inside the
affinity stage from `boltz2_aff.ckpt`), so both files are patched. The two pretrained files have identical structure-side weights.
Every key in the weights file must exist in the base state dict with the same shape; the affinity modules are never touched.
"""

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("weights", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "boltz")
    args = parser.parse_args()

    tuned = torch.load(args.weights, map_location="cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("boltz2_conf.ckpt", "boltz2_aff.ckpt"):
        checkpoint = torch.load(args.cache / name, map_location="cpu", weights_only=False)
        state = checkpoint["state_dict"]
        changed = 0
        for key, value in tuned.items():
            assert key in state and state[key].shape == value.shape, f"{name}: {key} missing or wrong shape"
            replacement = value.to(state[key].dtype)
            changed += int(not torch.equal(state[key], replacement))
            state[key] = replacement
        torch.save(checkpoint, args.out_dir / name)
        print(f"wrote {args.out_dir / name}: replaced {len(tuned)} tensors, {changed} differ from pretrained")


if __name__ == "__main__":
    main()

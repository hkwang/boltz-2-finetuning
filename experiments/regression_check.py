"""Regression check for Boltz-2 fine-tuning: co-fold deposited complexes and score the ligand pose and protein fold.

Run from the repository root, normally through experiments/run_regression.sh:
    python experiments/regression_check.py prepare          # inputs YAMLs; downloads missing CIFs (needs network)
    python experiments/regression_check.py make-checkpoint  # base checkpoint + fine-tuned tensors -> Boltz checkpoint
    python experiments/regression_check.py score --results experiments/results/regression/baseline
    python experiments/regression_check.py compare baseline finetuned
    python experiments/regression_check.py perturb --noise-scale 0.5 --output noisy_weights.pt   # positive control
"""

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd
import torch
import yaml

from boltz2_finetuning.regression import (
    compare_to_reference,
    perturbed_weights,
    polymer_sequence,
    predicted_ca,
    predicted_ligand,
    read_structure,
    reference_ca,
    reference_ligand,
    write_finetuned_checkpoint,
)

RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
RESULTS_ROOT = Path("experiments/results/regression")


def fetch_reference(pdb_id: str, structure_dir: Path) -> Path:
    path = structure_dir / f"{pdb_id}.cif"
    if not path.exists():
        structure_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(RCSB_CIF_URL.format(pdb_id=pdb_id), timeout=60).read())
    return path


def prepare(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text())
    args.input_dir.mkdir(parents=True, exist_ok=True)
    for system in config["systems"]:
        reference = read_structure(fetch_reference(system["pdb_id"], args.structure_dir))
        boltz_input = {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": polymer_sequence(reference, system["chain"]), "msa": config["msa"]}},
                {"ligand": {"id": "B", "ccd": system["ligand_ccd"]}},
            ],
        }
        path = args.input_dir / f"{system['name']}.yaml"
        path.write_text(yaml.safe_dump(boltz_input, sort_keys=False))
        print(f"{path}  {system['pdb_id']} chain {system['chain']} + {system['ligand_ccd']}")


def perturb(args: argparse.Namespace) -> None:
    weights = perturbed_weights(args.base.expanduser(), args.modules, args.noise_scale, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, args.output)
    print(f"wrote {args.output}: {len(weights)} tensors of {args.modules}, noise {args.noise_scale} x tensor std, seed {args.seed}")


def make_checkpoint(args: argparse.Namespace) -> None:
    count = write_finetuned_checkpoint(args.base.expanduser(), args.weights, args.output)
    print(f"wrote {args.output} with {count} fine-tuned tensors replaced")


def score(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text())
    predictions = args.results / "boltz_results_inputs" / "predictions"
    rows = []
    for system in config["systems"]:
        reference = read_structure(args.structure_dir / f"{system['pdb_id']}.cif")
        reference_cas = reference_ca(reference, system["chain"])
        reference_atoms = reference_ligand(reference, system["chain"], system["ligand_ccd"])
        for sample in range(config["predict"]["diffusion_samples"]):
            prefix = predictions / system["name"] / f"{system['name']}_model_{sample}"
            prediction = read_structure(prefix.with_suffix(".cif"))
            confidence = json.loads(Path(f"{predictions / system['name']}/confidence_{system['name']}_model_{sample}.json").read_text())
            rows.append(
                {
                    "system": system["name"],
                    "pdb_id": system["pdb_id"],
                    "sample": sample,
                    **compare_to_reference(reference_cas, reference_atoms, predicted_ca(prediction, "A"), predicted_ligand(prediction, "B")),
                    "confidence_score": confidence["confidence_score"],
                    "iptm": confidence["iptm"],
                    "ligand_iptm": confidence["ligand_iptm"],
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(args.results / "scores.csv", index=False)
    print(summarize(table, config["success_rmsd"]).to_string(float_format="{:.2f}".format))
    print(f"wrote {args.results / 'scores.csv'}")


def summarize(table: pd.DataFrame, success_rmsd: float) -> pd.DataFrame:
    """One row per system: the top-ranked sample (model_0) and the best of all samples."""
    top = table[table["sample"] == 0].set_index("system")
    best = table.groupby("system")[["ca_rmsd", "ligand_rmsd_pocket"]].min()
    return pd.DataFrame(
        {
            "ca_rmsd_top": top["ca_rmsd"],
            "lig_rmsd_pocket_top": top["ligand_rmsd_pocket"],
            "lig_rmsd_pocket_best": best["ligand_rmsd_pocket"],
            "lig_rmsd_global_top": top["ligand_rmsd_global"],
            "iptm_top": top["iptm"],
            "top_ok": top["ligand_rmsd_pocket"] < success_rmsd,
            "n_ok": table.assign(ok=table["ligand_rmsd_pocket"] < success_rmsd).groupby("system")["ok"].sum(),
        }
    )


def compare(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text())
    tables = {tag: summarize(pd.read_csv(RESULTS_ROOT / tag / "scores.csv"), config["success_rmsd"]) for tag in (args.baseline, args.finetuned)}
    base, tuned = tables[args.baseline], tables[args.finetuned]
    delta = (tuned[["ca_rmsd_top", "lig_rmsd_pocket_top", "lig_rmsd_pocket_best"]] - base[["ca_rmsd_top", "lig_rmsd_pocket_top", "lig_rmsd_pocket_best"]]).add_prefix("delta_")
    print(pd.concat({args.baseline: base, args.finetuned: tuned}, axis=1).to_string(float_format="{:.2f}".format))
    print("\nfinetuned - baseline (positive = worse):")
    print(delta.to_string(float_format="{:+.2f}".format))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/regression.yaml"))
    parser.add_argument("--structure-dir", type=Path, default=Path("data/structures/regression"))
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--input-dir", type=Path, default=Path("data/processed/regression/inputs"))
    prepare_parser.set_defaults(run=prepare)

    perturb_parser = commands.add_parser("perturb", help="positive control: noisy copy of the base diffusion weights")
    perturb_parser.add_argument("--base", type=Path, default=Path("~/.cache/boltz/boltz2_conf.ckpt"))
    perturb_parser.add_argument("--modules", nargs="+", default=["structure_module", "diffusion_conditioning"])
    perturb_parser.add_argument("--noise-scale", type=float, required=True, help="noise std as a fraction of each tensor's std")
    perturb_parser.add_argument("--seed", type=int, default=0)
    perturb_parser.add_argument("--output", type=Path, required=True)
    perturb_parser.set_defaults(run=perturb)

    checkpoint_parser = commands.add_parser("make-checkpoint")
    checkpoint_parser.add_argument("--base", type=Path, default=Path("~/.cache/boltz/boltz2_conf.ckpt"))
    checkpoint_parser.add_argument("--weights", type=Path, required=True, help="trainable_weights.pt from train_diffusion.py")
    checkpoint_parser.add_argument("--output", type=Path, required=True)
    checkpoint_parser.set_defaults(run=make_checkpoint)

    score_parser = commands.add_parser("score")
    score_parser.add_argument("--results", type=Path, required=True, help="boltz predict output directory")
    score_parser.set_defaults(run=score)

    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("baseline", help="tag under experiments/results/regression/")
    compare_parser.add_argument("finetuned")
    compare_parser.set_defaults(run=compare)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()

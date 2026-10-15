#!/usr/bin/env bash
# Co-fold the configs/regression.yaml complexes on Polaris and score them against the crystal structures.
# TAG (default baseline) names experiments/results/regression/<TAG>. Set WEIGHTS=<trainable_weights.pt> to run a fine-tuned
# model instead of stock boltz2_conf.ckpt; run once with the default TAG and once with WEIGHTS, then see the compare command.
# Set NOISE_SCALE=<x> instead to run the positive control: the diffusion weights of boltz2_conf.ckpt plus Gaussian noise of x times
# each tensor's std. The temporary weights and checkpoint (1-2 GB each) go to node-local /tmp and are deleted on exit.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$project_root/src"

cache_dir="${BOLTZ_CACHE:-$HOME/.cache/boltz}"
tag="${TAG:-baseline}"
output_dir="$project_root/experiments/results/regression/$tag"
input_dir="data/processed/regression/inputs"
mkdir -p "$output_dir"

for required_file in boltz2_conf.ckpt boltz2_aff.ckpt mols.tar; do
    [[ -s "$cache_dir/$required_file" ]] || { echo "Missing $cache_dir/$required_file" >&2; exit 2; }
done

echo "Host: $(hostname)"
nvidia-smi -L

python experiments/regression_check.py prepare --input-dir "$input_dir"

checkpoint_args=()
if [[ -n "${NOISE_SCALE:-}" || -n "${WEIGHTS:-}" ]]; then
    scratch="$(mktemp -d /tmp/regression.XXXXXX)"
    trap 'rm -rf "$scratch"' EXIT
    weights="${WEIGHTS:-$scratch/noisy_weights.pt}"
    if [[ -n "${NOISE_SCALE:-}" ]]; then
        python experiments/regression_check.py perturb --noise-scale "$NOISE_SCALE" --output "$weights"
    fi
    python experiments/regression_check.py make-checkpoint --weights "$weights" --output "$scratch/boltz2_finetuned.ckpt"
    checkpoint_args=(--checkpoint "$scratch/boltz2_finetuned.ckpt")
fi

read -r seed recycling sampling samples < <(python -c '
import yaml
p = yaml.safe_load(open("configs/regression.yaml"))["predict"]
print(p["seed"], p["recycling_steps"], p["sampling_steps"], p["diffusion_samples"])
')

SECONDS=0
boltz predict "$input_dir" \
    --out_dir "$output_dir" \
    --cache "$cache_dir" \
    --model boltz2 \
    "${checkpoint_args[@]}" \
    --accelerator gpu \
    --devices 1 \
    --recycling_steps "$recycling" \
    --sampling_steps "$sampling" \
    --diffusion_samples "$samples" \
    --num_workers 0 \
    --preprocessing-threads 4 \
    --seed "$seed" \
    --override
echo "boltz predict wall time: ${SECONDS}s"

python experiments/regression_check.py score --results "$output_dir"

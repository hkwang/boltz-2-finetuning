#!/usr/bin/env bash
# Co-fold Mac1 with the ligands in configs/mac1_cofold.yaml on Polaris A100s,
# using cuEquivariance kernels (Boltz default) and no MSA.
# DEVICES=N (default 1) runs Boltz data-parallel: each GPU predicts different ligands.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cache_dir="${BOLTZ_CACHE:-$HOME/.cache/boltz}"
input_dir="data/processed/mac1_cofold/inputs"
output_dir="${MAC1_COFOLD_OUTPUT:-$project_root/experiments/results/mac1_cofold}"
devices="${DEVICES:-1}"
mkdir -p "$output_dir"

for required_file in boltz2_conf.ckpt boltz2_aff.ckpt mols.tar; do
    [[ -s "$cache_dir/$required_file" ]] || { echo "Missing $cache_dir/$required_file" >&2; exit 2; }
done

echo "Host: $(hostname)"
nvidia-smi -L
python testing/boltz2_kernel_check.py

python experiments/prepare_mac1_cofold.py --out-dir "$input_dir"
read -r seed recycling sampling diffusion sampling_aff diffusion_aff < <(python -c '
import yaml
p = yaml.safe_load(open("configs/mac1_cofold.yaml"))["predict"]
print(p["seed"], p["recycling_steps"], p["sampling_steps"], p["diffusion_samples"],
      p["sampling_steps_affinity"], p["diffusion_samples_affinity"])
')

nvidia-smi dmon -s pucm -d 2 >"$output_dir/gpu-dmon.log" 2>&1 &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

# No --no_kernels flag: Boltz uses the cuEquivariance kernels by default.
SECONDS=0
boltz predict "$input_dir" \
    --out_dir "$output_dir" \
    --cache "$cache_dir" \
    --model boltz2 \
    --accelerator gpu \
    --devices "$devices" \
    --recycling_steps "$recycling" \
    --sampling_steps "$sampling" \
    --diffusion_samples "$diffusion" \
    --sampling_steps_affinity "$sampling_aff" \
    --diffusion_samples_affinity "$diffusion_aff" \
    --num_workers 0 \
    --preprocessing-threads 4 \
    --seed "$seed" \
    --override

echo "boltz predict wall time: ${SECONDS}s on ${devices} GPU(s)"
python experiments/summarize_mac1_cofold.py --results "$output_dir"

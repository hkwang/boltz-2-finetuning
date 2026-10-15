#!/usr/bin/env bash
# Co-fold the 45 held-out Mac1 entries (data/processed/mac1_eval/inputs) and score them with experiments/evaluate_mac1.py.
# LABEL names the run (output: experiments/results/mac1_eval/$LABEL). CKPT_DIR, if set, holds boltz2_conf.ckpt and
# boltz2_aff.ckpt written by experiments/make_finetuned_checkpoint.py; otherwise the pretrained weights in the Boltz cache are used.
# INPUTS_DIR and ENTRIES (defaults: all 45 held-out YAMLs and eval_entries.csv) select a subset, e.g. for smoke tests.
# DEVICES=N (default 4) runs Boltz data-parallel over entries. Predict settings follow configs/mac1_cofold.yaml except 5 structure samples.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$project_root/src"

label="${LABEL:?Set LABEL with qsub -v}"
cache_dir="${BOLTZ_CACHE:-$HOME/.cache/boltz}"
output_dir="$project_root/experiments/results/mac1_eval/$label"
devices="${DEVICES:-4}"
mkdir -p "$output_dir"

checkpoint_args=()
if [[ -n "${CKPT_DIR:-}" ]]; then
    checkpoint_args=(--checkpoint "$CKPT_DIR/boltz2_conf.ckpt" --affinity_checkpoint "$CKPT_DIR/boltz2_aff.ckpt")
fi

echo "Host: $(hostname) label=$label devices=$devices checkpoints=${CKPT_DIR:-pretrained}"
nvidia-smi -L
nvidia-smi dmon -s pucm -d 5 >"$output_dir/gpu-dmon.log" 2>&1 &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

read -r seed recycling sampling sampling_aff diffusion_aff < <(python -c '
import yaml
p = yaml.safe_load(open("configs/mac1_cofold.yaml"))["predict"]
print(p["seed"], p["recycling_steps"], p["sampling_steps"], p["sampling_steps_affinity"], p["diffusion_samples_affinity"])
')

SECONDS=0
boltz predict "${INPUTS_DIR:-data/processed/mac1_eval/inputs}" \
    --out_dir "$output_dir" \
    --cache "$cache_dir" \
    "${checkpoint_args[@]}" \
    --model boltz2 \
    --accelerator gpu \
    --devices "$devices" \
    --recycling_steps "$recycling" \
    --sampling_steps "$sampling" \
    --diffusion_samples 5 \
    --sampling_steps_affinity "$sampling_aff" \
    --diffusion_samples_affinity "$diffusion_aff" \
    --num_workers 0 \
    --preprocessing-threads 4 \
    --seed "$seed" \
    --override
echo "boltz predict wall time: ${SECONDS}s on ${devices} GPU(s)"

python experiments/evaluate_mac1.py "$output_dir" --label "$label" --entries "${ENTRIES:-data/processed/mac1_eval/eval_entries.csv}"

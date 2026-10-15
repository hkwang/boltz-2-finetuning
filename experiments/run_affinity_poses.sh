#!/usr/bin/env bash
# Predict one pose per affinity-training complex (data/processed/mac1_affinity/shard$SHARD) with `boltz predict`.
# The structure stage writes pre_affinity_<id>.npz (the best-ranked predicted complex), which experiments/cache_affinity_features.py crops and caches.
# CKPT_DIR (optional) holds boltz2_conf.ckpt and boltz2_aff.ckpt from make_finetuned_checkpoint.py; otherwise the pretrained weights are used.
# Re-running resumes: complexes with an existing prediction are skipped. The affinity stage is cut to 1 sample and 10 steps because only the pose is used here.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

shard="${SHARD:?Set SHARD with qsub -v}"
output_dir="${POSES_DIR:-$project_root/experiments/results/affinity_poses}/shard$shard"
checkpoint_args=()
if [[ -n "${CKPT_DIR:-}" ]]; then
    checkpoint_args=(--checkpoint "$CKPT_DIR/boltz2_conf.ckpt" --affinity_checkpoint "$CKPT_DIR/boltz2_aff.ckpt")
fi
mkdir -p "$output_dir"
echo "Host: $(hostname) shard=$shard checkpoints=${CKPT_DIR:-pretrained}"
nvidia-smi dmon -s pucm -d 30 >"$output_dir/gpu-dmon.log" 2>&1 &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

read -r seed recycling sampling < <(python -c '
import yaml
p = yaml.safe_load(open("configs/mac1_cofold.yaml"))["predict"]
print(p["seed"], p["recycling_steps"], p["sampling_steps"])
')

SECONDS=0
boltz predict "data/processed/mac1_affinity/shard$shard" \
    --out_dir "$output_dir" \
    --cache "${BOLTZ_CACHE:-$HOME/.cache/boltz}" \
    "${checkpoint_args[@]}" \
    --model boltz2 --accelerator gpu --devices "${DEVICES:-4}" \
    --recycling_steps "$recycling" --sampling_steps "$sampling" --diffusion_samples 1 \
    --sampling_steps_affinity 10 --diffusion_samples_affinity 1 \
    --num_workers 0 --preprocessing-threads 4 --seed "$seed"
echo "boltz predict wall time: ${SECONDS}s"
echo "POSES DONE shard $shard"

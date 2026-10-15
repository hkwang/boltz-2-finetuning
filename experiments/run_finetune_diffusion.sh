#!/usr/bin/env bash
# Diffusion fine-tuning on Polaris. CONFIG (default configs/finetune_diffusion.yaml) and extra key=value overrides in OVERRIDES.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$project_root/src"

echo "Host: $(hostname)"
nvidia-smi -L
python testing/boltz2_kernel_check.py
nvidia-smi dmon -s pucm -d 5 >"${DMON_LOG:-/dev/null}" 2>&1 &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

# shellcheck disable=SC2086
python experiments/train_diffusion.py "${CONFIG:-configs/finetune_diffusion.yaml}" ${OVERRIDES:-}

#!/usr/bin/env bash
# One fine-tuning run followed by held-out evaluation of intermediate weights, all in one PBS job (4 GPUs).
# Required env: NAME (run name, also the eval label prefix), CONFIG (configs/finetune_diffusion*.yaml).
# Optional: OVERRIDES (key=value pairs for train_diffusion.py), EVAL_STEPS (default "400 800 1400": which weights/step<N>.pt to score),
# OUT_ROOT (default /eagle/.../experiments/results/finetune_diffusion; the run lives in $OUT_ROOT/$NAME).
# Training resumes from $OUT_ROOT/$NAME/checkpoints/last.ckpt, so resubmitting the same job continues it.
# Each evaluation writes experiments/results/mac1_eval/${NAME}_step<N>/ (see run_mac1_eval.sh) and a Boltz checkpoint pair under $OUT_ROOT/$NAME/boltz_ckpt/step<N>.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root/src"

name="${NAME:?Set NAME with qsub -v}"
config="${CONFIG:?Set CONFIG with qsub -v}"
out_root="${OUT_ROOT:-/eagle/GenesisHackathonOct26/hkwang/boltz2-finetuning/experiments/results/finetune_diffusion}"
run_dir="$out_root/$name"

DMON_LOG="$out_root/dmon_$name.log" CONFIG="$config" OVERRIDES="name=$name output_dir=$out_root ${OVERRIDES:-}" bash experiments/run_finetune_diffusion.sh

for step in ${EVAL_STEPS:-400 800 1400}; do
    python experiments/make_finetuned_checkpoint.py "$run_dir/weights/step$step.pt" "$run_dir/boltz_ckpt/step$step"
    LABEL="${name}_step$step" CKPT_DIR="$run_dir/boltz_ckpt/step$step" bash experiments/run_mac1_eval.sh
done
echo "SWEEP DONE $name"

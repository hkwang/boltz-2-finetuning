#!/usr/bin/env bash
# One candidate end to end on this node's 4 GPUs: train -> export Boltz checkpoints -> score on validation at 3 seeds -> post to the board.
# Env: AGENT, NAME (run name; unique per candidate), METHOD (one line for the leaderboard) are required.
# Optional: CONFIG (default configs/finetune_diffusion.yaml), OVERRIDES (key=value for train_diffusion.py), NOTES, SEEDS, WEIGHTS (default the
# final trainable_weights.pt; or e.g. weights/step800.pt relative to the run directory).
# Commit your changes first: the result is recorded with the commit it came from, and the commit is pushed to the hub.
# Resubmitting the same NAME resumes training from its last.ckpt.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root/src"

agent="${AGENT:?Set AGENT}"
name="${NAME:?Set NAME}"
method="${METHOD:?Set METHOD}"
runs="$project_root/experiments/results/runs"
run_dir="$runs/$name"

python experiments/board.py check-frozen
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "commit your tracked changes before running a candidate" >&2
    exit 1
fi
git push -q origin "HEAD:refs/heads/$(git rev-parse --abbrev-ref HEAD)"

DMON_LOG="$runs/dmon_$name.log" CONFIG="${CONFIG:-configs/finetune_diffusion.yaml}" OVERRIDES="name=$name output_dir=$runs ${OVERRIDES:-}" \
    bash experiments/run_finetune_diffusion.sh

python experiments/make_finetuned_checkpoint.py "$run_dir/${WEIGHTS:-trainable_weights.pt}" "$run_dir/boltz_ckpt/final"
eval_output="$(LABEL="$name" CKPT_DIR="$run_dir/boltz_ckpt/final" SEEDS="${SEEDS:-2026 1 2}" bash experiments/run_val_eval.sh | tee /dev/stderr)"
eval_args="$(sed -n 's/^post-result args: //p' <<<"$eval_output")"

# shellcheck disable=SC2086
python experiments/board.py post-result --agent "$agent" --label "$name" --method "$method" --notes "${NOTES:-}" \
    --ckpt "$run_dir/boltz_ckpt/final" --train-run "$run_dir" $eval_args

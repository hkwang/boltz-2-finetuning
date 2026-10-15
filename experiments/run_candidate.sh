#!/usr/bin/env bash
# One candidate end to end on this node's 4 GPUs: train -> export Boltz checkpoints -> score on validation at 3 seeds -> post to the board.
# Env: AGENT, NAME (run name; unique per candidate), METHOD (one line for the leaderboard) are required.
# Optional: CONFIG (default configs/finetune_diffusion.yaml), OVERRIDES (key=value for train_diffusion.py), NOTES, SEEDS, WEIGHTS (default the
# final trainable_weights.pt; or e.g. weights/step800.pt relative to the run directory).
# Commit your changes first: the result is recorded with the commit it came from, and the commit is pushed to the hub.
# Resubmitting the same NAME resumes training from its last.ckpt. Each submission appends the commit, config text, overrides and job id to $run_dir/run_info.json (one JSON line per attempt).
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

mkdir -p "$run_dir"
python - "$run_dir/run_info.json" <<'PY'
import json
import os
import subprocess
import sys
import time

config = os.environ.get("CONFIG", "configs/finetune_diffusion.yaml")
git = lambda *args: subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
info = {
    "agent": os.environ["AGENT"], "name": os.environ["NAME"], "method": os.environ["METHOD"], "notes": os.environ.get("NOTES", ""),
    "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pbs_job": os.environ.get("PBS_JOBID"),
    "commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "config": config,
    "config_text": open(config).read(), "overrides": os.environ.get("OVERRIDES", ""), "weights": os.environ.get("WEIGHTS", "trainable_weights.pt"),
    "eval_seeds": os.environ.get("SEEDS", "2026 1 2"),
}
with open(sys.argv[1], "a") as handle:
    handle.write(json.dumps(info) + "\n")
PY

DMON_LOG="$runs/dmon_$name.log" CONFIG="${CONFIG:-configs/finetune_diffusion.yaml}" OVERRIDES="name=$name output_dir=$runs ${OVERRIDES:-}" \
    bash experiments/run_finetune_diffusion.sh

python experiments/make_finetuned_checkpoint.py "$run_dir/${WEIGHTS:-trainable_weights.pt}" "$run_dir/boltz_ckpt/final"
eval_output="$(LABEL="$name" CKPT_DIR="$run_dir/boltz_ckpt/final" SEEDS="${SEEDS:-2026 1 2}" bash experiments/run_val_eval.sh | tee /dev/stderr)"
eval_args="$(sed -n 's/^post-result args: //p' <<<"$eval_output")"

# shellcheck disable=SC2086
python experiments/board.py post-result --agent "$agent" --label "$name" --method "$method" --notes "${NOTES:-}" \
    --ckpt "$run_dir/boltz_ckpt/final" --train-run "$run_dir" $eval_args

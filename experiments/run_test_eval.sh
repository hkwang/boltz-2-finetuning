#!/usr/bin/env bash
# OPERATOR ONLY: score one candidate on the 24 held-out TEST entries (plus the 21 val entries) at three predict seeds, once, after the agents are done.
# Env: LABEL (required, e.g. final_<agent>_<run>), CKPT_DIR (boltz_ckpt/final of the candidate). Output: experiments/results/mac1_eval/test_${LABEL}_seed<S>/ and a summary line per seed.
# Agents must not run this: the validation-only score on the board is the only signal they may use.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
label="${LABEL:?Set LABEL}"
ckpt_dir="${CKPT_DIR:?Set CKPT_DIR}"
for seed in ${SEEDS:-2026 1 2}; do
    LABEL="test_${label}_seed${seed}" SEED="$seed" CKPT_DIR="$ckpt_dir" bash experiments/run_mac1_eval.sh
done
python experiments/compare_evals.py

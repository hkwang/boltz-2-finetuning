#!/usr/bin/env bash
# Score several (checkpoint, predict seed) combinations on the 45 held-out entries in one job, to measure sampling noise.
# SPECS is a space-separated list of LABEL:CKPT_DIR:SEED (CKPT_DIR empty = pretrained weights), e.g. "stock_seed1::1 ft_seed1:/path/boltz_ckpt/step1400:1".
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

for spec in ${SPECS:?Set SPECS with qsub -v}; do
    IFS=: read -r label ckpt_dir seed <<<"$spec"
    LABEL="$label" CKPT_DIR="$ckpt_dir" SEED="$seed" bash experiments/run_mac1_eval.sh
done

#!/usr/bin/env bash
# Score a Boltz-2 checkpoint pair on the 21 validation entries (the 24 test entries are never predicted) at several predict seeds.
# Env: LABEL (required); CKPT_DIR (boltz2_conf.ckpt + boltz2_aff.ckpt from experiments/make_finetuned_checkpoint.py; unset = pretrained);
# SEEDS (default "2026 1 2": the seed of configs/mac1_cofold.yaml plus two more, matching the stock baseline on the board); DEVICES (default 4).
# Outputs experiments/results/mac1_eval/val_${LABEL}_seed<S>/ and prints the --eval arguments for `experiments/board.py post-result`.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

label="${LABEL:?Set LABEL}"
inputs="experiments/results/val_inputs"
if [[ ! -f "$inputs/eval_entries.csv" ]]; then
    mkdir -p "$inputs/inputs"
    python - "$inputs" <<'PY'
import shutil
import sys
from pathlib import Path

import pandas as pd

out = Path(sys.argv[1])
entries = pd.read_csv("data/processed/mac1_eval/eval_entries.csv")
val = entries[entries["split"] == "val"]
for pdb_id in val["pdb_id"]:
    shutil.copy(f"data/processed/mac1_eval/inputs/mac1_eval_{pdb_id}.yaml", out / "inputs")
val.to_csv(out / "eval_entries.csv", index=False)
print(f"{len(val)} validation entries")
PY
fi

eval_args=()
for seed in ${SEEDS:-2026 1 2}; do
    run="val_${label}_seed${seed}"
    LABEL="$run" SEED="$seed" CKPT_DIR="${CKPT_DIR:-}" INPUTS_DIR="$inputs/inputs" ENTRIES="$inputs/eval_entries.csv" bash experiments/run_mac1_eval.sh
    eval_args+=("${seed}:$project_root/experiments/results/mac1_eval/$run")
done
echo "post-result args: --eval ${eval_args[*]}"

#!/usr/bin/env bash
# End-to-end smoke test with all outputs on /eagle: full fine-tune and LoRA fine-tune (6 steps, 4 GPUs), LoRA resume from last.ckpt,
# weights -> Boltz checkpoints, and a 2-entry held-out co-fold + scoring with the LoRA checkpoint. Run via jobs/polaris/run_job.pbs.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
eagle_root="${EAGLE_ROOT:-/eagle/GenesisHackathonOct26/hkwang/boltz2-finetuning}"
smoke="$eagle_root/experiments/results/smoke"
mkdir -p "$smoke"

common="trainer.devices=4 trainer.max_steps=6 trainer.checkpoint_every_n_steps=3 trainer.save_weights_every_n_steps=3 output_dir=$smoke"
export DMON_LOG="$smoke/dmon.log"

OVERRIDES="name=full $common" bash experiments/run_finetune_diffusion.sh
CONFIG=configs/finetune_diffusion_lora.yaml OVERRIDES="name=lora $common" bash experiments/run_finetune_diffusion.sh
CONFIG=configs/finetune_diffusion_lora.yaml OVERRIDES="name=lora $common trainer.max_steps=8" bash experiments/run_finetune_diffusion.sh

for name in full lora; do
    ls -la "$smoke/$name/weights" "$smoke/$name/checkpoints"
done
python experiments/make_finetuned_checkpoint.py "$smoke/lora/trainable_weights.pt" "$smoke/lora_ckpt"

subset="$smoke/eval_inputs"
mkdir -p "$subset"
cp data/processed/mac1_eval/inputs/mac1_eval_13RI.yaml data/processed/mac1_eval/inputs/mac1_eval_13RJ.yaml "$subset"
python - "$smoke/entries.csv" <<'PY'
import sys
import pandas as pd
table = pd.read_csv("data/processed/mac1_eval/eval_entries.csv")
table[table["pdb_id"].isin(["13RI", "13RJ"])].to_csv(sys.argv[1], index=False)
PY
LABEL=smoke_lora CKPT_DIR="$smoke/lora_ckpt" DEVICES=1 INPUTS_DIR="$subset" ENTRIES="$smoke/entries.csv" bash experiments/run_mac1_eval.sh
echo "SMOKE OK"

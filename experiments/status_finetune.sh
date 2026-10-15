#!/usr/bin/env bash
# Status of the wave-1 fine-tuning jobs: queue, latest step, mean diffusion loss. Usage: bash experiments/status_finetune.sh [run names...]
cd "$(dirname "${BASH_SOURCE[0]}")/.."
date -u +%H:%M; qstat -u $USER | tail -n +4 | awk '{print $1,$4,$10,$11}'
runs="${*:-full_lr1e-5 full_lr3e-6 lora_r16_lr1e-4}"
for j in $runs; do f=$(ls -t $j.o* 2>/dev/null | head -1)
echo "== $j: $(grep -ao 'step [0-9]* record [A-Z0-9]* tokens [0-9]* atoms [0-9]* loss [0-9.]* [0-9.]*s peak [0-9.]*GB' $f | tail -1)"
grep -aE "Traceback|Error|SWEEP DONE" $f | head -2; done
for r in $runs; do ~/venvs/boltz2/bin/python - $r <<'PY'
import sys,pandas as pd,glob
f=glob.glob(f"experiments/results/finetune_diffusion/{sys.argv[1]}/logs/version_*/metrics.csv")
if f:
    d=pd.read_csv(sorted(f)[-1]); d=d[d["train/diffusion_loss"].notna()] if "train/diffusion_loss" in d else d
    if len(d): print(sys.argv[1],"steps",int(d.step.max()),"loss first50 %.3f last50 %.3f"%(d["train/diffusion_loss"].head(50).mean(),d["train/diffusion_loss"].tail(50).mean()))
PY
done

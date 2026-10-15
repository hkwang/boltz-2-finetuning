#!/usr/bin/env bash
set -euo pipefail

echo "Host: $(hostname)"
echo "PBS_JOBID=${PBS_JOBID:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "--- nvidia-smi -L"
nvidia-smi -L

echo "--- per-GPU query"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv

python - <<'PY' &
import time
import torch

x = torch.randn(8192, 8192, device="cuda")
end = time.time() + 15
while time.time() < end:
    x = x @ x
    x = x / x.norm()
torch.cuda.synchronize()
PY
load_pid=$!
sleep 8

echo "--- compute apps while a CUDA process runs"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

echo "--- dmon, 3 samples"
nvidia-smi dmon -s pucvmet -c 3

echo "--- process table"
ps -u "$USER" -o pid,etime,%cpu,%mem,cmd --sort=-%cpu | head -4

wait "$load_pid"
echo "GPU query check finished"

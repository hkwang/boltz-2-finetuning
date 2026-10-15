#!/usr/bin/env bash
# Load GPUs 0-2 for 100 s and leave GPU 3 idle, so the GPU monitor and its
# summary can be checked against a known pattern. Run through jobs/polaris/run_job.pbs.
set -euo pipefail

for gpu in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$gpu python - <<'PY' &
import time
import torch

x = torch.randn(8192, 8192, device="cuda")
end = time.time() + 100
while time.time() < end:
    x = x @ x
    x = x / x.norm()
torch.cuda.synchronize()
PY
done
wait
echo "gpu monitor load test finished"

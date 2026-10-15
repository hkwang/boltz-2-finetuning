#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cache_dir="${BOLTZ_CACHE:-$HOME/.cache/boltz}"
output_dir="${BOLTZ_SMOKE_OUTPUT:-$project_root/testing/results/boltz2_cuda_smoke}"
mkdir -p "$cache_dir" "$output_dir"

for required_file in boltz2_conf.ckpt boltz2_aff.ckpt mols.tar; do
    if [[ ! -s "$cache_dir/$required_file" ]]; then
        echo "Missing Boltz-2 cache file: $cache_dir/$required_file" >&2
        echo "Stage Boltz-2 checkpoints and CCD data from a login node before submitting." >&2
        exit 2
    fi
done

echo "Host: $(hostname)"
echo "Boltz cache: $cache_dir"
echo "Output directory: $output_dir"
nvidia-smi -L

python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this PBS allocation")

device_count = torch.cuda.device_count()
print(f"CUDA devices visible to PyTorch: {device_count}")
for device_id in range(device_count):
    print(f"GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
PY

gpu_metrics="$output_dir/gpu-dmon.log"
nvidia-smi dmon -s pucvmet -d 2 >"$gpu_metrics" 2>&1 &
monitor_pid=$!
stop_gpu_monitor() {
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
}
trap stop_gpu_monitor EXIT

boltz predict testing/boltz2_cuda_smoke.yaml \
    --out_dir "$output_dir" \
    --cache "$cache_dir" \
    --model boltz2 \
    --accelerator gpu \
    --devices 1 \
    --recycling_steps 1 \
    --sampling_steps 10 \
    --diffusion_samples 1 \
    --sampling_steps_affinity 10 \
    --diffusion_samples_affinity 1 \
    --num_workers 0 \
    --preprocessing-threads 1 \
    --seed 2026 \
    --override

if ! find "$output_dir/boltz_results_boltz2_cuda_smoke" -type f \
    \( -name '*.cif' -o -name '*.pdb' \) -print -quit | grep -q .; then
    echo "Boltz finished without writing a coordinate prediction" >&2
    exit 1
fi

echo "Boltz-2 CUDA smoke prediction succeeded. GPU metrics: $gpu_metrics"

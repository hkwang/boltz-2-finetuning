#!/usr/bin/env bash
# Sample nvidia-smi every GPU_MONITOR_INTERVAL seconds (default 30) until killed.
# Usage: gpu_monitor.sh OUTPUT_PREFIX
# Writes OUTPUT_PREFIX.gpu.csv (one row per GPU per sample) and
# OUTPUT_PREFIX.apps.csv (one row per compute process per sample).
set -uo pipefail

prefix="${1:?Usage: gpu_monitor.sh OUTPUT_PREFIX}"
interval="${GPU_MONITOR_INTERVAL:-30}"

gpu_fields=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu
app_fields=timestamp,gpu_uuid,pid,process_name,used_memory

echo "$gpu_fields" >"$prefix.gpu.csv"
echo "$app_fields" >"$prefix.apps.csv"

while true; do
    nvidia-smi --query-gpu="$gpu_fields" --format=csv,noheader,nounits >>"$prefix.gpu.csv"
    nvidia-smi --query-compute-apps="$app_fields" --format=csv,noheader,nounits >>"$prefix.apps.csv"
    sleep "$interval"
done

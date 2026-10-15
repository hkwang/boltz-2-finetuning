#!/usr/bin/env bash
# Keep a compute node allocated for interactive testing over ssh: sleeps for HOLD_SECONDS (default 3 h). Run via jobs/polaris/run_job.pbs.
set -euo pipefail
echo "Host: $(hostname)"
sleep "${HOLD_SECONDS:-10800}"

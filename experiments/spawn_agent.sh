#!/usr/bin/env bash
# Create an agent's clone on /eagle and submit its 4xA100 worker job (jobs/polaris/run_worker.pbs). Run on a Polaris login node (swarm/launch.sh does this over ssh).
# Usage: bash experiments/spawn_agent.sh AGENT_NAME [walltime HH:MM:SS, default 08:00:00]
# Env: SWARM_ROOT (default /eagle/GenesisHackathonOct26/hkwang/boltz2-swarm), VENV_DIR (default ~/venvs/boltz2).
set -euo pipefail

agent="${1:?agent name (letters, digits, - and _)}"
walltime="${2:-08:00:00}"
swarm_root="${SWARM_ROOT:-/eagle/GenesisHackathonOct26/hkwang/boltz2-swarm}"
venv_dir="${VENV_DIR:-$HOME/venvs/boltz2}"
agent_dir="$swarm_root/agents/$agent"
repo="$agent_dir/repo"
[[ "$agent" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "bad agent name" >&2; exit 1; }

mkdir -p "$agent_dir/results"
if [[ ! -d "$repo/.git" ]]; then
    git clone -q "$swarm_root/repo.git" "$repo"
    git -C "$repo" config user.name "$agent"
    git -C "$repo" config user.email "$agent@swarm.local"
    ln -sfn "$swarm_root/shared/structures" "$repo/data/structures"
    ln -sfn "$swarm_root/shared/boltz_inputs" "$repo/data/processed/boltz_inputs"
    ln -sfn "$agent_dir/results" "$repo/experiments/results"
fi
rm -f "$agent_dir/stop"

seconds=$(( $(echo "$walltime" | awk -F: '{print $1*3600+$2*60+$3}') - 900 ))
TMPDIR=/tmp qsub -N "worker_$agent" -l walltime="$walltime" \
    -v "AGENT=$agent,SWARM_ROOT=$swarm_root,VENV_DIR=$venv_dir,WORKER_SECONDS=$seconds" \
    -o "$agent_dir/worker.out" "$repo/jobs/polaris/run_worker.pbs"

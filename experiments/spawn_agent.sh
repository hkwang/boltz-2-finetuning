#!/usr/bin/env bash
# Create (or reuse) an agent's clone on /eagle and submit its 4xA100 PBS job. Run from a login node.
# Usage: bash experiments/spawn_agent.sh AGENT_NAME "one-line brief" [walltime HH:MM:SS, default 08:00:00]
# Env: SWARM_ROOT (default /eagle/GenesisHackathonOct26/hkwang/boltz2-swarm), VENV_DIR (default ~/venvs/boltz2), AGENT_MODEL, AGENT_HOURS (hours the agent may work, default walltime - 1).
set -euo pipefail

agent="${1:?agent name (letters, digits, - and _)}"
brief="${2:?one-line brief}"
walltime="${3:-08:00:00}"
swarm_root="${SWARM_ROOT:-/eagle/GenesisHackathonOct26/hkwang/boltz2-swarm}"
venv_dir="${VENV_DIR:-$HOME/venvs/boltz2}"
hub="$swarm_root/repo.git"
agent_dir="$swarm_root/agents/$agent"
repo="$agent_dir/repo"
[[ "$agent" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "bad agent name" >&2; exit 1; }

mkdir -p "$agent_dir"
if [[ ! -d "$repo/.git" ]]; then
    git clone -q "$hub" "$repo"
    git -C "$repo" checkout -q -b "agent/$agent"
    git -C "$repo" config user.name "$agent"
    git -C "$repo" config user.email "$agent@swarm.local"
    mkdir -p "$repo/data/processed" "$repo/experiments/results"
    ln -sfn "$swarm_root/shared/structures" "$repo/data/structures"
    ln -sfn "$swarm_root/shared/boltz_inputs" "$repo/data/processed/boltz_inputs"
    ln -sfn "$agent_dir/results" "$repo/experiments/results"
    mkdir -p "$agent_dir/results"
fi

hours="${AGENT_HOURS:-$(( ${walltime%%:*} - 1 ))}"
TMPDIR=/tmp qsub -N "agent_$agent" -l walltime="$walltime" \
    -v "AGENT=$agent,BRIEF=$brief,SWARM_ROOT=$swarm_root,VENV_DIR=$venv_dir,AGENT_HOURS=$hours,AGENT_MODEL=${AGENT_MODEL:-gpt-6.1-sol}" \
    -o "$agent_dir/pbs.out" "$repo/jobs/polaris/run_agent.pbs"

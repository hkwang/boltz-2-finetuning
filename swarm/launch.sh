#!/usr/bin/env bash
# Start the agents on this machine, one tmux session each, plus a Polaris worker node per agent.
# Usage: swarm/launch.sh [AGENT ...]     (default: every agent in swarm/briefs.tsv; names are the first column)
# Env: LOCAL_ROOT (default ~/boltz2-swarm), POLARIS_USER (default hkwang), CODEX_ARGS (default selects model gpt-6.1-sol via provider mag),
#      SWARM_HOURS (agent working time, default 7.5), WALLTIME (worker job, default 08:00:00), NO_WORKER=1 to skip qsub.
# First run swarm/connect.sh in another terminal. Needs: ssh, git, tmux, python3, codex (logged in / AMSC_TOKEN set in the environment).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
user="${POLARIS_USER:-hkwang}"
host="${POLARIS_HOST:-polaris.alcf.anl.gov}"
root_remote="${SWARM_ROOT:-/eagle/GenesisHackathonOct26/hkwang/boltz2-swarm}"
local_root="${LOCAL_ROOT:-$HOME/boltz2-swarm}"
export CODEX_ARGS="${CODEX_ARGS:--c model_provider=mag -m gpt-6.1-sol}"
ssh_options=(-o ControlMaster=auto -o "ControlPath=$HOME/.ssh/swarm-%C" -o ControlPersist=12h)
export GIT_SSH_COMMAND="ssh ${ssh_options[*]}"

ssh -O check -o "ControlPath=$HOME/.ssh/swarm-%C" "$user@$host" 2>/dev/null || { echo "no open ssh connection: run swarm/connect.sh first" >&2; exit 1; }
command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }
[[ -n "${AMSC_TOKEN:-}" ]] || echo "warning: AMSC_TOKEN is not set in this shell; agents need whatever credentials your codex uses" >&2

mkdir -p "$local_root"
selected=("$@")
while IFS=$'\t' read -r agent brief; do
    [[ -z "$agent" || "$agent" == \#* ]] && continue
    if [[ ${#selected[@]} -gt 0 && ! " ${selected[*]} " =~ " $agent " ]]; then continue; fi
    work="$local_root/$agent/repo"
    if [[ ! -d "$work/.git" ]]; then
        mkdir -p "$local_root/$agent"
        git clone -q "ssh://$user@$host$root_remote/repo.git" "$work"
        git -C "$work" config user.name "$agent"
        git -C "$work" config user.email "$agent@swarm.local"
        git -C "$work" checkout -q -b "agent/$agent"
    fi
    if [[ "${NO_WORKER:-0}" != 1 ]]; then
        SWARM_AGENT="$agent" bash "$here/polaris.sh" restart "${WALLTIME:-08:00:00}" || echo "warning: could not submit the worker for $agent" >&2
    fi
    if tmux has-session -t "swarm-$agent" 2>/dev/null; then
        echo "tmux session swarm-$agent already exists; leaving it running"
        continue
    fi
    tmux new-session -d -s "swarm-$agent" -c "$work" \
        "SWARM_AGENT='$agent' SWARM_BRIEF=$(printf %q "$brief") SWARM_HOURS='${SWARM_HOURS:-7.5}' CODEX_ARGS='$CODEX_ARGS' bash '$here/agent_loop.sh'; echo; echo 'agent loop ended; press enter'; read"
    echo "started $agent (tmux attach -t swarm-$agent; log $local_root/$agent/.swarm-logs/$agent.log)"
done <"$here/briefs.tsv"
echo "watch the team: swarm/watch.sh"

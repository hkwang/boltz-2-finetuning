#!/usr/bin/env bash
# Stop all agents: kill their tmux sessions. Polaris jobs keep running; add --cancel to cancel each agent's active job too.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py="${SWARM_PY:-$HOME/boltz2-swarm-tools/.venv/bin/python}"
for session in $(tmux ls -F '#S' 2>/dev/null | grep '^swarm-' || true); do
    agent="${session#swarm-}"
    tmux kill-session -t "$session" && echo "stopped $agent"
    if [[ "${1:-}" == --cancel ]]; then
        SWARM_AGENT="$agent" "$py" "$here/iri.py" status | awk '$3 == "queued" || $3 == "active" || $3 == "running" {print $1}' |
            while read -r job; do SWARM_AGENT="$agent" "$py" "$here/iri.py" cancel "$job"; done
    fi
done

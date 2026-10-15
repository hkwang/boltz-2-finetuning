#!/usr/bin/env bash
# Stop all agents: kill the tmux sessions and tell each Polaris worker to exit (running jobs finish first; qdel them yourself to free nodes now).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for session in $(tmux ls -F '#S' 2>/dev/null | grep '^swarm-'); do
    agent="${session#swarm-}"
    tmux kill-session -t "$session" && echo "stopped $agent"
    SWARM_AGENT="$agent" bash "$here/polaris.sh" sh "touch \$SWARM_ROOT/agents/$agent/stop" || true
done

#!/usr/bin/env bash
# Operator view of the whole team: leaderboard, latest messages, agents and each agent's latest Polaris jobs.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="${SWARM_ROOT:-$HOME/boltz2-swarm}"
py="${SWARM_PY:-$HOME/boltz2-swarm-tools/.venv/bin/python}"
export SWARM_ROOT="$root"
echo "=== leaderboard"; SWARM_AGENT=operator "$py" "$here/iri.py" board leaderboard --top 15 || true
echo; echo "=== latest messages"; SWARM_AGENT=operator "$py" "$here/iri.py" board read --tail 15 || true
echo; echo "=== jobs"
for jobs in "$root"/agents/*/jobs.jsonl; do
    [[ -e "$jobs" ]] || continue
    agent="$(basename "$(dirname "$jobs")")"
    echo "-- $agent"; SWARM_AGENT="$agent" "$py" "$here/iri.py" status | tail -n 2 || true
done

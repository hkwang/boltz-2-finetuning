#!/usr/bin/env bash
# Start N persistent agents on this laptop, one tmux session and one git clone each. Polaris is reached through swarm/iri.py (IRI + Globus).
# Usage: swarm/launch.sh [N | AGENT ...]     (default 9 agents named agent1..agent9)
# Env: SWARM_ROOT (default ~/boltz2-swarm: hub repo, board, clones), SWARM_PY (python with pandas/scipy/httpx/globus-sdk, default
#      ~/boltz2-swarm-tools/.venv/bin/python), IRI_REPO (gm_hackathon_agentic_ai clone holding mcp/ and .env), CODEX_ARGS, SWARM_HOURS (default 7.5).
# Needs: tmux, git, codex (AMSC_TOKEN exported), Globus Connect Personal running, valid ALCF IRI + Globus tokens in $IRI_REPO/.env.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(dirname "$here")"
root="${SWARM_ROOT:-$HOME/boltz2-swarm}"
py="${SWARM_PY:-$HOME/boltz2-swarm-tools/.venv/bin/python}"
export CODEX_ARGS="${CODEX_ARGS:--c model_provider=mag -m gpt-6.1-sol}"
command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }
[[ -x "$py" ]] || { echo "no python at $py (set SWARM_PY)" >&2; exit 1; }
[[ -n "${AMSC_TOKEN:-}" ]] || echo "warning: AMSC_TOKEN is not set in this shell; agents need it to reach the model" >&2
[[ -f "$root/board/frozen.json" ]] || { echo "no board at $root/board (see swarm/README.md, 'Before the first launch')" >&2; exit 1; }

if [[ "${SKIP_E2E:-0}" != 1 ]]; then
    tested="$(cat "$root/e2e_passed" 2>/dev/null || true)"
    [[ -n "$tested" ]] || { echo "run swarm/e2e_test.sh first (or SKIP_E2E=1): no passing end-to-end check recorded" >&2; exit 1; }
    git -C "$repo_root" diff --quiet "$tested" HEAD -- swarm experiments src configs jobs ||
        { echo "pipeline code changed since the last passing swarm/e2e_test.sh (${tested:0:7}); rerun it (or SKIP_E2E=1)" >&2; exit 1; }
fi
if [[ ! -d "$root/repo.git" ]]; then
    git clone -q --bare "$repo_root" "$root/repo.git"
fi
git -C "$repo_root" push -q "$root/repo.git" HEAD:refs/heads/master

if [[ $# -eq 0 ]]; then set -- 9; fi
if [[ $# -eq 1 && "$1" =~ ^[0-9]+$ ]]; then
    count="$1"; set --
    for index in $(seq 1 "$count"); do set -- "$@" "agent$index"; done
fi

for agent in "$@"; do
    [[ "$agent" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "bad agent name $agent" >&2; exit 1; }
    work="$root/$agent/repo"
    if [[ ! -d "$work/.git" ]]; then
        mkdir -p "$root/$agent"
        git clone -q "$root/repo.git" "$work"
        git -C "$work" config user.name "$agent"
        git -C "$work" config user.email "$agent@swarm.local"
        git -C "$work" checkout -q -b "agent/$agent"
    fi
    if tmux has-session -t "swarm-$agent" 2>/dev/null; then
        echo "tmux session swarm-$agent already exists; leaving it running"
        continue
    fi
    SWARM_ROOT="$root" SWARM_AGENT="$agent" "$py" "$here/iri.py" board register --agent "$agent" --deadline "$(date -u -d "+${SWARM_HOURS:-7.5} hours" +%FT%TZ)" >/dev/null
    tmux new-session -d -s "swarm-$agent" -c "$work" \
        "export PATH='$(dirname "$py")':\$PATH SWARM_ROOT='$root' SWARM_AGENT='$agent' SWARM_HOURS='${SWARM_HOURS:-7.5}' CODEX_ARGS='$CODEX_ARGS' IRI_REPO='${IRI_REPO:-$HOME/Documents/codex/myproject/gm_hackathon_agentic_ai}'; bash '$here/agent_loop.sh'; echo; echo 'agent loop ended; press enter'; read"
    echo "started $agent (tmux attach -t swarm-$agent; log $root/$agent/.swarm-logs/$agent.log)"
done
echo "watch the team: swarm/watch.sh"

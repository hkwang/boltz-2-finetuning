#!/usr/bin/env bash
# Runs one agent in the current directory until its deadline, resuming the same Codex session whenever the process exits early.
# Env: SWARM_AGENT, SWARM_HOURS (default 7.5), CODEX_ARGS (extra args, e.g. "-m gpt-6.1-sol -c model_provider=mag").
set -uo pipefail

agent="${SWARM_AGENT:?}"
hours="${SWARM_HOURS:-7.5}"
buffer="${SWARM_BUFFER_HOURS:-1}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
end=$(( $(date +%s) + $(awk "BEGIN{printf \"%d\", $hours*3600}") ))
deadline="$(date -u -d "@$end" +%Y-%m-%dT%H:%MZ)"
log_dir="$PWD/../.swarm-logs"
mkdir -p "$log_dir"
log="$log_dir/$agent.log"

prompt="$(python3 - "$here/agent_prompt.md" "$agent" "$deadline" "$buffer" <<'PY'
import sys
template, agent, deadline, buffer = sys.argv[1:]
print(open(template).read().replace("{AGENT}", agent).replace("{DEADLINE}", deadline).replace("{BUFFER}", buffer))
PY
)"

session=""
attempt=0
while [[ $(date +%s) -lt $end ]]; do
    attempt=$((attempt + 1))
    echo "=== $(date -u +%FT%TZ) attempt $attempt session ${session:-new}" | tee -a "$log"
    if [[ -z "$session" ]]; then
        # shellcheck disable=SC2086
        codex exec --json --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check ${CODEX_ARGS:-} "$prompt" 2>>"$log.err" | tee -a "$log.jsonl" | python3 -u "$here/show_events.py" "$log"
    else
        # shellcheck disable=SC2086
        codex exec resume --json --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check ${CODEX_ARGS:-} "$session" \
            "You stopped before the deadline ($deadline UTC; now $(date -u +%H:%MZ)). Re-read docs/swarm.md if needed, check the board (swarm/iri.py board leaderboard; board read --agent $agent --unread) and swarm/iri.py status, and continue your work." \
            2>>"$log.err" | tee -a "$log.jsonl" | python3 -u "$here/show_events.py" "$log"
    fi
    session="$(python3 - "$log.jsonl" <<'PY'
import json, sys
thread = ""
for line in open(sys.argv[1]):
    try:
        event = json.loads(line)
    except ValueError:
        continue
    if event.get("type") == "thread.started":
        thread = event["thread_id"]
        break
print(thread)
PY
)"
    sleep 20
done
echo "=== $(date -u +%FT%TZ) deadline reached; agent $agent finished" | tee -a "$log"

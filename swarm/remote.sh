#!/usr/bin/env bash
# Runs on a Polaris login node, piped over ssh by swarm/polaris.sh: remote.sh OP AGENT [args]. Env: SWARM_ROOT, VENV_DIR.
# Login nodes are shared: this script only moves files, calls qsub/qstat and runs the light board commands.
set -uo pipefail
export PATH="$PATH:/opt/pbs/bin"

op="${1:?op}"
agent="${2:?agent}"
shift 2
root="${SWARM_ROOT:?}"
venv="${VENV_DIR:-$HOME/venvs/boltz2}"
dir="$root/agents/$agent"
repo="$dir/repo"

resolve_id() {
    local want="$1" found
    if [[ "$want" == latest ]]; then
        found="$(ls -t "$dir/logs" 2>/dev/null | grep '\.log$' | head -n 1)"
    else
        found="$(ls "$dir/logs" "$dir/queue" 2>/dev/null | grep -F "$want" | sed 's/\.\(log\|sh\|exit\|pid\|start\)$//' | sort -u | tail -n 1)"
    fi
    echo "${found%.log}"
}

case "$op" in
spawn)
    git --git-dir="$root/repo.git" show master:experiments/spawn_agent.sh >"/tmp/spawn_agent.$$.sh"
    SWARM_ROOT="$root" VENV_DIR="$venv" bash "/tmp/spawn_agent.$$.sh" "$agent" "${1:-08:00:00}"
    rm -f "/tmp/spawn_agent.$$.sh"
    ;;
enqueue)
    name="$1" sha="$2" payload="$3"
    [[ -d "$repo/.git" ]] || { echo "no clone for $agent on Polaris; run swarm/launch.sh first" >&2; exit 2; }
    [[ "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "bad job name (use letters, digits, . _ -)" >&2; exit 2; }
    id="$(date +%Y%m%d-%H%M%S)-$name"
    mkdir -p "$dir/queue"
    {
        echo '#!/usr/bin/env bash'
        echo 'set -eo pipefail'
        echo "cd $repo"
        [[ "$sha" != none ]] && echo "git fetch -q origin && git checkout -q -B agent/$agent $sha"
        echo 'echo "== $(date -u +%FT%TZ) $(hostname) commit $(git rev-parse --short HEAD)"'
        echo "$payload" | base64 -d
    } >"$dir/queue/.$id.tmp"
    mv "$dir/queue/.$id.tmp" "$dir/queue/$id.sh"
    echo "$id"
    ;;
status)
    jobs="$(qselect -u "$USER" -N "worker_$agent" -s QRH 2>/dev/null)"
    if [[ -n "$jobs" ]]; then
        # shellcheck disable=SC2086
        qstat -fw $jobs 2>/dev/null | grep -E "Job_Id|job_state|resources_used.walltime|Resource_List.walltime|exec_host" | sed 's/^ *//'
    else
        echo "NO WORKER JOB: run 'swarm/polaris.sh restart' (your queued commands will not run without it)"
    fi
    echo "-- worker: $(tail -n 1 "$dir/worker.txt" 2>/dev/null)"
    echo "-- queued:"; ls "$dir/queue" 2>/dev/null | sed 's/\.sh$//' | sed 's/^/   /'
    echo "-- running:"
    for file in "$dir"/running/*.sh; do
        [[ -e "$file" ]] || continue
        id="$(basename "$file" .sh)"
        echo "   $id (running $(( ($(date +%s) - $(cat "$dir/logs/$id.start" 2>/dev/null || date +%s)) / 60 )) min)"
    done
    echo "-- finished (latest 8):"
    for file in $(ls -t "$dir"/logs/*.exit 2>/dev/null | head -n 8); do
        id="$(basename "$file" .exit)"
        echo "   $id exit $(cat "$file")"
    done
    ;;
log)
    id="$(resolve_id "${1:-latest}")"
    [[ -n "$id" ]] || { echo "no such job"; exit 1; }
    echo "== $id ($(cat "$dir/logs/$id.exit" 2>/dev/null | sed 's/^/exit /;s/^$/no exit yet/'))"
    tail -n "${2:-60}" "$dir/logs/$id.log" 2>/dev/null | cut -c1-400
    ;;
wait)
    id="$(resolve_id "${1:?job id}")"
    limit="${2:-1200}"
    waited=0
    while [[ ! -e "$dir/logs/$id.exit" && $waited -lt $limit ]]; do
        sleep 10
        waited=$((waited + 10))
    done
    if [[ -e "$dir/logs/$id.exit" ]]; then echo "$id finished with exit $(cat "$dir/logs/$id.exit")"; else echo "$id still running/queued after ${limit}s"; fi
    tail -n 15 "$dir/logs/$id.log" 2>/dev/null | cut -c1-300
    ;;
cancel)
    id="$(resolve_id "${1:?job id}")"
    if [[ -e "$dir/queue/$id.sh" ]]; then
        rm "$dir/queue/$id.sh" && echo "removed $id from the queue"
    elif [[ -e "$dir/running/$id.sh" ]]; then
        touch "$dir/kill-$id"
        echo "kill requested for $id; the worker stops it within about 5 s (check with log)"
    else
        echo "$id is not queued or running"
    fi
    ;;
sh)
    cd "$repo" && timeout 120 bash -c "$(echo "$1" | base64 -d)"
    ;;
board)
    [[ -d "$repo/.git" ]] || git clone -q "$root/repo.git" "$repo"
    git -C "$repo" fetch -q origin 2>/dev/null
    [[ "$agent" == operator ]] && git -C "$repo" merge -q --ff-only origin/master 2>/dev/null
    cd "$repo" && SWARM_ROOT="$root" AGENT="$agent" "$venv/bin/python" experiments/board.py "${@}"
    ;;
*)
    echo "unknown op $op" >&2
    exit 2
    ;;
esac

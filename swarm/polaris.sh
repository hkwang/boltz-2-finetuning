#!/usr/bin/env bash
# The agent's interface to Polaris, run from your laptop. All agents share one ssh connection (see swarm/connect.sh).
# Needs env: SWARM_AGENT (your agent name). Optional: POLARIS_USER, POLARIS_HOST, SWARM_ROOT, VENV_DIR.
#
#   run NAME 'COMMAND'    push your HEAD (commit first), queue COMMAND on your 4xA100 node; prints the job id
#   status                worker job state, queued/running/finished jobs
#   log [ID] [LINES]      tail a job log (default the latest job, 60 lines)
#   wait ID [SECONDS]     block until the job ends (default 1200 s), then print its exit code and last lines
#   cancel ID             drop a queued job or kill a running one
#   sh 'COMMAND'          run a short command (120 s limit) in your clone on the login node, e.g. ls, cat, tail
#   get REMOTE LOCAL      copy a path relative to your agent directory on /eagle to LOCAL (rsync)
#   board ARGS...         experiments/board.py: leaderboard | read --unread | say --kind note "text" | agents | post-result ...
#   restart [HH:MM:SS]    submit a new worker job if yours ended (default 08:00:00)
set -euo pipefail

agent="${SWARM_AGENT:?SWARM_AGENT is not set (start it through swarm/launch.sh)}"
host="${POLARIS_HOST:-polaris.alcf.anl.gov}"
user="${POLARIS_USER:-hkwang}"
export SWARM_ROOT="${SWARM_ROOT:-/eagle/GenesisHackathonOct26/hkwang/boltz2-swarm}"
venv="${VENV_DIR:-/home/${REMOTE_HOME_USER:-hkwang}/venvs/boltz2}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ssh_options=(-o ControlMaster=auto -o "ControlPath=$HOME/.ssh/swarm-%C" -o ControlPersist=12h -o ServerAliveInterval=60 -o ServerAliveCountMax=10)

export GIT_SSH_COMMAND="ssh ${ssh_options[*]}"

remote() {
    local quoted=""
    for argument in "$@"; do quoted+=" $(printf %q "$argument")"; done
    ssh "${ssh_options[@]}" "$user@$host" "SWARM_ROOT=$SWARM_ROOT VENV_DIR=$venv bash -s --$quoted" <"$here/remote.sh"
}
encode() { printf %s "$1" | base64 | tr -d '\n'; }

command="${1:?usage: polaris.sh run|status|log|wait|cancel|sh|get|board|restart}"
shift
case "$command" in
run)
    name="${1:?job name}"
    shift
    [[ $# -ge 1 ]] || { echo "usage: polaris.sh run NAME COMMAND" >&2; exit 2; }
    if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
        echo "commit your changes first (git add -A && git commit -m ...): the job runs the committed code" >&2
        exit 1
    fi
    branch="agent/$agent"
    git push -q origin "HEAD:refs/heads/$branch" --force
    remote enqueue "$agent" "$name" "$(git rev-parse HEAD)" "$(encode "$*")"
    ;;
status) remote status "$agent" ;;
log) remote log "$agent" "${1:-latest}" "${2:-60}" ;;
wait) remote wait "$agent" "${1:?job id}" "${2:-1200}" ;;
cancel) remote cancel "$agent" "${1:?job id}" ;;
sh) remote sh "$agent" "$(encode "$*")" ;;
board) remote board "$agent" "$@" ;;
restart) remote spawn "$agent" "${1:-08:00:00}" ;;
get)
    rsync -az -e "ssh ${ssh_options[*]}" "$user@$host:$SWARM_ROOT/agents/$agent/${1:?remote path}" "${2:?local path}"
    ;;
*) echo "unknown command $command" >&2; exit 2 ;;
esac

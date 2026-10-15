#!/usr/bin/env bash
# Operator view of the whole team: leaderboard, latest messages, agents and worker jobs. Run from your laptop (needs the open ssh connection).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SWARM_AGENT=operator
echo "=== leaderboard"; bash "$here/polaris.sh" board leaderboard --top 15
echo; echo "=== latest messages"; bash "$here/polaris.sh" board read --tail 15
echo; echo "=== agents"; bash "$here/polaris.sh" board agents
echo; echo "=== your PBS jobs"; ssh -o ControlMaster=auto -o "ControlPath=$HOME/.ssh/swarm-%C" "${POLARIS_USER:-hkwang}@${POLARIS_HOST:-polaris.alcf.anl.gov}" 'export PATH=$PATH:/opt/pbs/bin; qstat -u $USER | tail -n +4'

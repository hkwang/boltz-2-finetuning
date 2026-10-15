#!/usr/bin/env bash
# Open the shared ssh connection to Polaris once (enter your password/token here); every agent reuses it for the next 12 h.
# Run this in its own terminal tab before swarm/launch.sh. Re-run it if swarm/polaris.sh starts asking for a password.
# Env: POLARIS_USER (default $USER), POLARIS_HOST (default polaris.alcf.anl.gov).
set -euo pipefail
host="${POLARIS_HOST:-polaris.alcf.anl.gov}"
user="${POLARIS_USER:-hkwang}"
mkdir -p ~/.ssh
if ssh -O check -o "ControlPath=$HOME/.ssh/swarm-%C" "$user@$host" 2>/dev/null; then
    echo "connection to $host is already open"
    exit 0
fi
ssh -fN -o ControlMaster=yes -o "ControlPath=$HOME/.ssh/swarm-%C" -o ControlPersist=12h -o ServerAliveInterval=60 "$user@$host"
ssh -o "ControlPath=$HOME/.ssh/swarm-%C" "$user@$host" 'echo connected to $(hostname) as $USER'

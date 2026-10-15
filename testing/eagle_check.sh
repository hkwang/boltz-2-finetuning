#!/usr/bin/env bash
# Compute-node check that the project directory on /eagle is readable and writable, and that the Boltz cache there is complete.
# EAGLE_ROOT (default /eagle/GenesisHackathonOct26/hkwang/boltz2-finetuning) is the root of the moved project data.
set -euo pipefail
root="${EAGLE_ROOT:-/eagle/GenesisHackathonOct26/hkwang/boltz2-finetuning}"
echo "Host: $(hostname)"
df -h "$root" | tail -1
ls "$root"
probe="$root/.eagle_check_$$"
dd if=/dev/zero of="$probe" bs=1M count=512 oflag=direct 2>&1 | tail -1
dd if="$probe" of=/dev/null bs=1M iflag=direct 2>&1 | tail -1
rm "$probe"
for required in ${REQUIRED_FILES:-}; do
    [[ -s "$required" ]] && echo "ok $required" || { echo "MISSING $required" >&2; exit 2; }
done

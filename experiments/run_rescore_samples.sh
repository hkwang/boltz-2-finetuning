#!/usr/bin/env bash
# Per-sample rescoring of every stored validation prediction in the swarm (agents/*/results/mac1_eval/val_*_seed*) and the
# stock runs, in parallel; one CSV per results directory under $SWARM_REMOTE/rescore/. Env: SWARM_REMOTE, JOBS (default 32), LIMIT (smoke test).
set -euo pipefail
out="$SWARM_REMOTE/rescore"
mkdir -p "$out"
stock=/eagle/GenesisHackathonOct26/hkwang/boltz2-finetuning/experiments/results/mac1_eval
{ ls -d "$stock"/stock "$stock"/stock_seed1 "$stock"/stock_seed2; ls -d "$SWARM_REMOTE"/agents/agent*/results/mac1_eval/val_*_seed*; } \
    | head -n "${LIMIT:-100000}" \
    | xargs -P "${JOBS:-32}" -I{} bash -c 'dir="$1"; tag=$(echo "$dir" | sed "s#.*/agents/##; s#/results/mac1_eval/#__#; s#.*/mac1_eval/#operator__#"); [ -s "$2/$tag.csv" ] || python experiments/rescore_samples.py "$2/$tag.csv" "$dir" >/dev/null 2>"$2/$tag.err" || echo "failed $dir"' _ {} "$out"
echo "RESCORE DONE: $(ls "$out"/*.csv | wc -l) tables"

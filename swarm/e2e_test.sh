#!/usr/bin/env bash
# Deterministic end-to-end check of the agent pipeline, no LLM involved: 50 training steps -> Boltz checkpoint export -> validation at the
# 3 predict seeds -> post-result, all through swarm/iri.py exactly as an agent would run it.
# Runs in a scratch root (E2E_ROOT, default ~/boltz2-swarm-e2e) with its own hub and a copy of the real board's frozen hashes and stock
# baselines, so the real leaderboard is untouched. On success writes $SWARM_ROOT/e2e_passed (the tested commit); swarm/launch.sh requires it.
# Usage: swarm/e2e_test.sh   (from a clean checkout; MINUTES=walltime, default 60)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(dirname "$here")"
real_root="${SWARM_ROOT:-$HOME/boltz2-swarm}"
test_root="${E2E_ROOT:-$HOME/boltz2-swarm-e2e}"
py="${SWARM_PY:-$HOME/boltz2-swarm-tools/.venv/bin/python}"
[[ -z "$(git -C "$repo_root" status --porcelain --untracked-files=no)" ]] || { echo "commit your changes first: the test runs HEAD" >&2; exit 1; }
[[ -f "$real_root/board/frozen.json" ]] || { echo "no board at $real_root/board" >&2; exit 1; }
commit="$(git -C "$repo_root" rev-parse HEAD)"
stamp="$(date -u +%Y%m%dT%H%M%S)"
name="e2e_$stamp"

mkdir -p "$test_root/board/baseline" "$test_root/board/cursors"
cp "$real_root/board/frozen.json" "$test_root/board/"
cp "$real_root"/board/baseline/*.csv "$test_root/board/baseline/"
[[ -s "$test_root/board/results.jsonl" ]] || head -n 1 "$real_root/board/results.jsonl" >"$test_root/board/results.jsonl"
touch "$test_root/board/messages.jsonl" "$test_root/board/agents.jsonl"
[[ -d "$test_root/repo.git" ]] || git clone -q --bare "$repo_root" "$test_root/repo.git"
git -C "$repo_root" push -q -f "$test_root/repo.git" HEAD:refs/heads/master
clone="$test_root/$name/repo"
git clone -q "$test_root/repo.git" "$clone"
cd "$clone"
git checkout -q -b agent/e2e
export SWARM_ROOT="$test_root" SWARM_AGENT=e2e PATH="$(dirname "$py"):$PATH"

echo "== $(date -u +%T) e2e $name at commit ${commit:0:7}"
python swarm/iri.py run "$name" --minutes "${MINUTES:-60}" \
    "AGENT=\$AGENT NAME=$name METHOD=e2e CONFIG=configs/finetune_diffusion.yaml OVERRIDES=\"seed=0 trainer.max_steps=50 trainer.checkpoint_every_n_steps=50 trainer.save_weights_every_n_steps=50\" bash experiments/run_candidate.sh"
python swarm/iri.py wait "$name" --every 60 --max 7200 --lines 20
python swarm/iri.py log "$name" --lines 200 >"$test_root/$name/job.log"
if ! grep -q "CANDIDATE DONE $name" "$test_root/$name/job.log"; then
    echo "FAIL: the job did not reach CANDIDATE DONE; log: $test_root/$name/job.log" >&2
    exit 1
fi
python swarm/iri.py post-result "$name" --job "$name" --method "e2e pipeline check, 50 steps"
python swarm/iri.py board leaderboard
grep -q "\"label\": \"$name\"" "$test_root/board/results.jsonl" || { echo "FAIL: no board row for $name" >&2; exit 1; }
echo "$commit" >"$real_root/e2e_passed"
echo "PASS: e2e $name at commit ${commit:0:7} (marker $real_root/e2e_passed)"

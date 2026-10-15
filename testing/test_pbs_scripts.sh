#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dummy_output="$(bash "$repo_root/testing/dummy_job.pbs")"
if [[ "$dummy_output" != *"dummy job succeeded"* ]]; then
    printf 'Expected dummy PBS job output, received: %s\n' "$dummy_output" >&2
    exit 1
fi

output="$(
    cd /tmp
    PBS_O_WORKDIR="$repo_root" \
        JOB_SCRIPT=testing/dummy_job.pbs \
        bash "$repo_root/jobs/polaris/run_job.pbs"
)"

if [[ "$output" != *"dummy job succeeded"* ]]; then
    printf 'Expected dummy job output, received: %s\n' "$output" >&2
    exit 1
fi

if PBS_O_WORKDIR="$repo_root" bash "$repo_root/jobs/polaris/run_job.pbs" \
    > /dev/null 2>&1; then
    printf 'Expected missing JOB_SCRIPT to fail\n' >&2
    exit 1
fi

printf 'PBS wrapper smoke checks passed\n'

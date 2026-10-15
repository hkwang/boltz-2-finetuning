# Polaris PBS jobs

`run_job.pbs` is a reusable Polaris GPU job wrapper. When `VENV_DIR` is set, it
loads `cray-python/3.12.12` and `cuda/12.9`, then activates that Python virtual
environment. The CUDA module provides runtime libraries required by Boltz's
cuEquivariance kernels. It then runs a project-owned shell script named by
`JOB_SCRIPT`. This keeps Boltz-specific commands in the project script, where
the actual training interface is known. Polaris does not provide a `conda`
module in the current login environment.

The PBS account line uses `GenesisHackathonOct26` and the queue is
`R7728155`, the Genesis allocation queue. Confirm this allocation is enabled
for your ALCF account before submitting. Adjust the walltime and GPU count to
the workload. The reusable job defaults request one Polaris node with four
GPUs for 30 minutes.

From the repository root, submit with:

```bash
qsub -v JOB_SCRIPT="$PWD/testing/dummy_job.pbs" \
  jobs/polaris/run_job.pbs
```

This submits the included dummy task; replace its path with your experiment's
shell entry point for a real run. The dummy file also has PBS headers and can be
submitted directly with `qsub testing/dummy_job.pbs`. For a real run, sync the
locked environment with `uv` and pass its path as `VENV_DIR`:

```bash
module load cray-python/3.12.12
UV_PROJECT_ENVIRONMENT="$HOME/venvs/boltz2" uv sync --python "$(which python)"
qsub -v JOB_SCRIPT="$PWD/experiments/run_training.sh",VENV_DIR="$HOME/venvs/boltz2" \
  jobs/polaris/run_job.pbs
```

`VENV_DIR` is optional if `JOB_SCRIPT` activates another environment itself.
The script path can be absolute or relative to `PBS_O_WORKDIR` (the submission
directory). Polaris job output is joined and written by the wrapper (not
PBS) to `logs/<jobname>.o<jobid>` under the submission directory, so it can be
followed live with `tail -f`; `logs/` is git-ignored and created at job start.

Polaris compute nodes do not have outbound network access. Before Boltz's first
prediction, download its two Boltz-2 checkpoints and CCD archive from a login
node into the shared cache (the default is `$HOME/.cache/boltz`):

```bash
cache="$HOME/.cache/boltz"
mkdir -p "$cache"
curl -fL https://model-gateway.boltz.bio/boltz2_conf.ckpt \
  -o "$cache/boltz2_conf.ckpt"
curl -fL https://model-gateway.boltz.bio/boltz2_aff.ckpt \
  -o "$cache/boltz2_aff.ckpt"
curl -fL https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar \
  -o "$cache/mols.tar"
```

The first prediction extracts `mols.tar` into the same cache. The included
`testing/boltz2_cuda_smoke.sh` checks that these files are staged, confirms
PyTorch can see the four allocated A100s, and runs a small GPU prediction on one
device. Its sample input sets `msa: empty` and does not call an MSA server. Submit
it from the project root with:

```bash
TMPDIR=/tmp qsub -l walltime=01:00:00 \
  -v JOB_SCRIPT="$PWD/testing/boltz2_cuda_smoke.sh",VENV_DIR="$HOME/venvs/boltz2" \
  jobs/polaris/run_job.pbs
```

The wrapper exports `CC=gcc` with the venv because Polaris's default `CC=nvc` cannot JIT-compile the Triton helper used by
the cuEquivariance kernels. Mac1 co-folding is documented in [`docs/mac1_cofold.md`](../../docs/mac1_cofold.md).

## GPU load monitoring

`run_job.pbs` starts `jobs/polaris/gpu_monitor.sh` in the background before `JOB_SCRIPT` and stops it when the job ends. It samples
`nvidia-smi` every 30 s (override with `-v GPU_MONITOR_INTERVAL=<seconds>`; `0` disables it) and writes two CSVs to `logs/` under the submission directory:
`gpu-monitor.<jobid>.gpu.csv` (per GPU: utilization, memory, power, temperature) and `gpu-monitor.<jobid>.apps.csv` (compute processes per GPU).
Both can be read from the login node while the job runs. Summarize them with:

```bash
~/venvs/boltz2/bin/python experiments/summarize_gpu_monitor.py logs/gpu-monitor.<jobid>
```

It prints mean/max utilization, peak memory, mean power and PIDs per GPU. It exits 1 and prints a warning if a GPU with a process averaged below
`--idle-pct` (default 10 %) utilization, or if a GPU ran no process while others did, which is the usual sign of a multi-GPU run using fewer GPUs than allocated.
Checked with `testing/gpu_monitor_load.sh` (job 7746175, 4 GPUs, load on GPUs 0-2): GPUs 0-2 averaged 92 %, GPU 3 was flagged.
Averages include startup and data-loading time, so judge a training run on the steady-state samples in the CSV, not only the mean.

To exercise the job wrapper without PBS or a GPU, run:

```bash
bash testing/test_pbs_scripts.sh
```

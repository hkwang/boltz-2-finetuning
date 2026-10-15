# Getting Boltz-2 running on Polaris: log

Chronological record of what was needed to run Boltz-2 on a Polaris A100 with cuEquivariance kernels and no MSA
(2026-10-14). Job logs are `boltz2-*.o*` and `mac1-cofold.o*` in the repo root. Items marked *(inferred)* come from
the repo state and job logs, not from watching the step happen.

## Working configuration

| Item | Value |
|---|---|
| PBS | account `GenesisHackathonOct26`, queue `R7728155`, `-l filesystems=home:eagle`, `select=1:ngpus=1` (or 4) |
| Modules at runtime | `cray-python/3.12.12`, `cuda/12.9` |
| Python env | `uv sync` -> `~/venvs/boltz2`; `boltz[cuda]==2.2.1`, `torch==2.11.0+cu128` (pinned in `pyproject.toml`, `uv.lock`) |
| Weights/data | `boltz2_conf.ckpt`, `boltz2_aff.ckpt`, `mols.tar` staged in `~/.cache/boltz` from a login node |
| Env vars | `CC=gcc` (set by `jobs/polaris/run_job.pbs`), `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` |
| Boltz flags | `--model boltz2 --accelerator gpu --devices 1 --num_workers 0 --preprocessing-threads N`; no `--no_kernels` |
| MSA | `msa: empty` per protein in the input YAML (single-sequence mode) |

## Steps and problems

1. **PBS wrapper.** `jobs/polaris/run_job.pbs` runs a `JOB_SCRIPT`, loading the modules and activating `VENV_DIR`. Checked with
   dummy jobs (`boltz2-dummy.o*`, `boltz2-job.o7745024`) before using Boltz. Submit with `TMPDIR=/tmp qsub ...`; the sandbox
   `/var/tmp` was read-only for PBS staging.
2. **Environment.** `uv sync` with the locked `pyproject.toml`. The cu128 PyTorch wheel runs on the Polaris CUDA 13 driver, and
   `boltz[cuda]` installs the cuEquivariance cu12 kernels. `cuda/12.9` must be loaded at runtime so cuEquivariance can find
   `libnvrtc.so.12` and cuBLAS.
3. **Offline compute nodes.** Compute nodes have no outbound network. Job `7745176` failed with
   `OSError: [Errno 101] Network is unreachable` from `urllib` while Boltz tried to fetch something *(inferred: a missing cache
   file; the log does not name it)*. Fix: download the checkpoints and `mols.tar` on a login node into `~/.cache/boltz`
   (commands in `jobs/polaris/README.md`), and make the run script fail early if they are missing
   (`testing/boltz2_cuda_smoke.sh` does).
4. **Smoke test.** `testing/boltz2_cuda_smoke.sh` (job `7745181`) predicts an 18-residue peptide + ethanol with 1 recycling
   step and 10 sampling steps. It succeeded: structure + affinity outputs on GPU 0, CUDA visible to PyTorch on all 4 GPUs.
   The log had a cuEquivariance "FALLING BACK to the slow PyTorch reference path" warning.
5. **Is that warning a problem?** It names the Blackwell SM100f kernel, which does not exist for A100 (compute capability 8.0),
   and notes short sequences fall back by design (`CUEQ_TRIATTN_FALLBACK_THRESHOLD`/`CUEQ_TRIMUL_FALLBACK_THRESHOLD`, default 100).
   The smoke protein is below 100 tokens, so it proved CUDA works but not that the kernels do. Added
   `testing/boltz2_kernel_check.py`, which calls the triangle ops on 256 tokens under the torch profiler and lists the CUDA kernels.
6. **Triton compile failure (job `7745692`).** The kernel check died with `nvc-Error-Unknown switch: -Wno-psabi`.
   Cause: Polaris's environment has `CC=/opt/nvidia/hpc_sdk/.../nvc`. Triton (used by cuEquivariance's layer-norm kernel)
   honours `$CC` and JIT-compiles a small CUDA helper with it, adding `-Wno-psabi`, a gcc-only flag
   (`triton/runtime/build.py`). Fix: `export CC=gcc` in `jobs/polaris/run_job.pbs` after activating the venv.
7. **Kernel check passes (job `7745696`).** On the A100: `fused_sigmoid_gated_dual_gemm_forward_kernel`,
   `layer_norm_transpose_forward_single_pass_kernel` and a CUTLASS sm80 bf16 GEMM for the triangle multiplicative update; a cuDNN
   `cudnn_sm80_fprop_sdpa` fused-attention kernel for triangle attention. No fallback warnings.
8. **Mac1 co-fold.** Six ligands ran with kernels and `msa: empty`, no failures (`docs/mac1_cofold.md`).
9. **Summary script path bug.** `boltz predict <directory>` writes to `boltz_results_<directory name>/`, here
   `boltz_results_inputs/`, not one folder per input. Fixed in `experiments/summarize_mac1_cofold.py`.

## Checks for a new setup

- `nvidia-smi -L` lists the A100s. `qstat -f <id>` shows `Resource_List.select` and `exec_vnode` with the requested `ngpus`.
- `python testing/boltz2_kernel_check.py` on a GPU node prints cuEquivariance kernel names and `fallback_warning=False`.
- Boltz output has a `.cif`, `confidence_*.json` and `affinity_*.json` per input.
- `gpu-dmon.log` from the smoke test shows the SM utilization while the job runs.

## Not verified

- The full Boltz run was not profiled, so I did not show which layers used the kernels or what speedup they give over `--no_kernels`.
- Multi-GPU (`--devices 4`) prediction was not tried. All runs used one GPU.
- A100 gets a cuDNN attention kernel, not the SM100f one. The predictions use bf16 AMP (Boltz default).

## Multi-GPU (4x A100) prediction

Verified 2026-10-14, job `7745741` (`select=1:ngpus=4`, exit 0, 1 min 49 s wall, `exec_vnode ngpus=4`).

**How it works.** `boltz predict --devices N` (N > 1) switches Lightning to `DDPStrategy` (`boltz/main.py`, fork start method). It is data parallel:
every rank holds a full Boltz-2 copy and predicts different inputs, with no model sharding and no speedup for a single input.
Boltz caps `N` at the number of inputs. The structure and affinity passes each build their own trainer, so both run on all GPUs.
No `torchrun`, `mpiexec` or NCCL setup was needed on a single node.

**Run.** `experiments/run_mac1_cofold.sh` takes `DEVICES` (default 1) and writes `gpu-dmon.log` to the output directory:

```bash
TMPDIR=/tmp qsub -l select=1:ngpus=4 -l walltime=00:30:00 -N mac1-cofold-4gpu \
  -v JOB_SCRIPT="$PWD/experiments/run_mac1_cofold.sh",VENV_DIR="$HOME/venvs/boltz2",DEVICES=4,MAC1_COFOLD_OUTPUT="$PWD/experiments/results/mac1_cofold_4gpu" \
  jobs/polaris/run_job.pbs
```

**Evidence.**
- Log shows `GLOBAL_RANK` 0-3, `MEMBER: 1/4`..`4/4` and four `LOCAL_RANK` lines; `Number of failed examples: 0`.
- `gpu-dmon.log`: all four GPUs reached 100% SM utilization (GPUs 0/1 busy in 22/21 of 50 samples, GPUs 2/3 in 12/13).
- Each rank's progress bar shows `2/2` batches. That is 8 slots for 6 ligands, so Lightning's distributed sampler most likely padded 2 ligands with repeats (not checked in the outputs).

**Timing (6 ligands, 200 sampling steps).**

| Pass | 1 GPU (job `7745696`) | 4 GPUs (job `7745741`) |
|---|---|---|
| Structure | 56 s | 18-22 s |
| Affinity | 64 s | 18-22 s |
| `boltz predict` total | not timed | 98 s (includes ~50 s of startup, preprocessing and checkpoint loading) |

Per-pass speedup is about 3x, not 4x: with 6 inputs on 4 ranks every rank runs 2 batches, so the ideal is 6/2 = 3x. Startup dominates at this size,
so a larger ligand set (a multiple of 4) is needed to measure real scaling.

**Results match to sampling noise.** 4-GPU vs 1-GPU affinity values differ by 0.03-0.17 (log10 uM) and iPTM by up to 0.06 (7HCO: 0.97 vs 0.91).
Ranks seem to use different random streams even with the same `--seed`; I did not confirm this. Ligand ranking by affinity is unchanged for 5 of 6, so treat single-seed values as noisy.

**Gotchas.**
- PBS `select=1:ngpus=4` leaves `CUDA_VISIBLE_DEVICES` unset, and Lightning reports `[0,1,2,3]`. With `ngpus=1` the node still lists all four GPUs in `nvidia-smi -L`, and the earlier "1 GPU" runs used device 0 only because of `--devices 1`. Use `exec_vnode` to confirm what PBS allocated.
- `nvidia-smi dmon` in `gpu-dmon.log` reports the node's GPUs, not only the ones your process uses.
- Single node only. Multi-node DDP (`--num_nodes`, `mpiexec` and a rendezvous setup) was not tried.

## Notes for training work (not tested)

I only ran inference. What I read in the installed `boltz==2.2.1` source:
- `boltz.model.models.boltz2.Boltz2` defines `training_step`, `validation_step`, `configure_optimizers` and an `ema` option. Its constructor default is `use_kernels=False`; `boltz predict` passes `use_kernels=not no_kernels`. A training script has to set `use_kernels=True` itself to get the cuEquivariance kernels.
- `Boltz2.setup()` only falls back to no kernels on pre-sm80 GPUs at predict time, so A100s are fine.
- Training data modules live in `boltz/data/module/training.py` and `trainingv2.py`. I did not check that the kernel backward pass works on A100 or that Boltz-2 training runs with these modules; Boltz ships no training CLI.
- The `CC=gcc` fix in `jobs/polaris/run_job.pbs` applies to training too, since it is needed for the Triton JIT.
- DDP on one node needs no extra launcher for predict. A Lightning training job can use the same `DDPStrategy`, but that is untested here.

## Monitoring GPUs with nvidia-smi (tested 2026-10-14)

`nvidia-smi` only sees the GPUs of the node it runs on, so it must run on the job's compute node.

**From the login node: not available.** `ssh <exec_host>` was denied (`Permission denied (publickey,keyboard-interactive,hostbased)`) while job 7746129 was running on `x3008c0s37b0n0`.
Find the node with `qstat -f <job-id> | grep -E 'job_state|exec_host|exec_vnode'`. Do not rely on SSH; use the in-job methods below.

**Inside the job (works).** `testing/gpu_query_check.sh` was run as job 7746141 (`select=1:ngpus=1`, 34 s walltime, `Exit_status = 0`) and printed, with a 15 s CUDA matmul loop running on GPU 0:

| Purpose | Command | Result in the test |
|---|---|---|
| Node inventory | `nvidia-smi -L` | 4 x A100-SXM4-40GB, even with `ngpus=1`; `CUDA_VISIBLE_DEVICES` unset |
| Per-GPU snapshot | `nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv` | CSV, one row per GPU |
| Which process uses a GPU | `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` | the test `python` PID with 1036 MiB |
| Utilization, power, clocks, memory | `nvidia-smi dmon -s pucvmet -c 3` | GPU 0: 100 % sm, ~295 W, 1047 MB; GPUs 1-3 idle (0 %, ~52 W, 5 MB) |
| Elapsed time and CPU of your processes | `ps -u "$USER" -o pid,etime,%cpu,%mem,cmd --sort=-%cpu \| head` | test process at ~150 % CPU |

**Logging for later (login-node readable).** `testing/boltz2_cuda_smoke.sh` runs `nvidia-smi dmon -s pucvmet -d 2 > .../gpu-dmon.log &` in the background and kills it on exit.
Job 7746129 (Boltz-2 smoke prediction, 1m19s, `Exit_status = 0`) produced 120 dmon rows in `testing/results/boltz2_cuda_smoke/gpu-dmon.log`; GPU 0 showed 1-36 % sm and 0.5-2.7 GB `fb` during the prediction.
Read it from the login node while the job runs: `tail -n 8 testing/results/boltz2_cuda_smoke/gpu-dmon.log`.

**Caveats.**
- Each `dmon` row is one GPU at one instant. `sm` of 0-36 % over a 2 s sampling interval on this tiny input says little about a real run.
- The node lists all 4 GPUs, so only the GPU holding your PID in `--query-compute-apps` is yours; judge by that, not by which GPUs are idle.
- Job state and exit status come from the login node: `qstat -f <job-id>`, or `qstat -xf <job-id>` after it finishes.

**Automatic load monitoring (2026-10-14).** `jobs/polaris/run_job.pbs` now samples `nvidia-smi` every 30 s into `gpu-monitor.<jobid>.{gpu,apps}.csv`;
see `jobs/polaris/README.md`. Test job 7746175 (`select=1:ngpus=4`, 2m07s, `Exit_status = 0`, 10 s interval) with load on GPUs 0-2 gave 92 % mean utilization, 288-302 W and 1047 MiB on those GPUs;
GPU 3 stayed at 0 % and 52 W and was flagged by `experiments/summarize_gpu_monitor.py`.

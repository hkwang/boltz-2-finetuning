# Mac1 co-folding with Boltz-2 on Polaris

Zero-shot Boltz-2 structure + affinity prediction for Mac1 (SARS-CoV-2 NSP3 macrodomain) with
ligands from `data/processed/mac1_splits.csv`, on one Polaris A100, using cuEquivariance kernels and no MSA.
Verified 2026-10-14 (PBS job 7745696, `x3008c0s13b0n0`, 1 GPU, 3 min 15 s wall time).

## Files

| File | Role |
|---|---|
| `configs/mac1_cofold.yaml` | Protein sequence, `msa: empty`, ligand PDB IDs, `boltz predict` settings |
| `experiments/prepare_mac1_cofold.py` | Writes one Boltz input YAML per ligand into `data/processed/mac1_cofold/inputs/` |
| `experiments/run_mac1_cofold.sh` | Kernel check, input prep, `boltz predict`, summary (PBS `JOB_SCRIPT`) |
| `experiments/summarize_mac1_cofold.py` | Joins predictions to experimental IC50 -> `experiments/results/mac1_cofold/summary.csv` |
| `testing/boltz2_kernel_check.py` | Profiles which CUDA kernels cuEquivariance launches on the visible GPU |

## Setup choices

- **Protein:** the 169-residue Mac1 construct (`SMVNSF...SSFL`), chain A, identical in RCSB 5RS7, 7HC4 and 13RI.
  The crystal asymmetric unit has two copies; only one is co-folded.
- **No MSA:** `msa: empty` on the protein. Boltz runs single-sequence mode and never contacts an MSA server
  (compute nodes have no network anyway). Predictions are expected to be worse than with an MSA.
- **Ligands:** six `test`-split pairs spanning ~0.02-1000 uM: 9CY0, 13RR, 7HC4, 7HEC, 7HCF, 7HCO (7HCO is censored, `>1000 uM`).
  SMILES come from the splits table. Edit `ligands:` in the config to change the set. Note that the test split is meant
  to stay held out from fine-tuning, so use it for zero-shot baselines only.
- **Settings:** seed 2026, 3 recycling steps, 200 sampling steps, 1 structure sample, 3 affinity samples; `--devices 1`.
- **Affinity:** `affinity_pred_value` is Boltz's log10(IC50 in uM) estimate (lower = tighter); `affinity_probability_binary` is
  the binder probability. Experimental `ic50_uM` is kept in the units of the source tables.

## Run

```bash
TMPDIR=/tmp qsub -l select=1:ngpus=1 -l walltime=01:00:00 -N mac1-cofold \
  -v JOB_SCRIPT="$PWD/experiments/run_mac1_cofold.sh",VENV_DIR="$HOME/venvs/boltz2" \
  jobs/polaris/run_job.pbs
```

The Boltz cache (`boltz2_conf.ckpt`, `boltz2_aff.ckpt`, `mols.tar`) must already be staged; see `jobs/polaris/README.md`.
Output: `logs/mac1-cofold.o<jobid>` under the submission directory and `experiments/results/mac1_cofold/`
(CIFs, confidence and affinity JSONs per ligand, `summary.csv`).

## Verification: CUDA kernels

Boltz uses the cuEquivariance kernels by default (`--no_kernels` is not passed). `testing/boltz2_kernel_check.py` runs on a
256-token pair tensor and prints the CUDA kernels launched, with no cuEquivariance fallback warning:

- `triangle_multiplicative_update`: `fused_sigmoid_gated_dual_gemm_forward_kernel`, `layer_norm_transpose_forward_single_pass_kernel`, CUTLASS sm80 bf16 GEMM.
- `triangle_attention`: `cudnn_fmha::fwd::...cudnn_sm80_fprop_sdpa` (cuEquivariance routes A100/sm80 to a cuDNN fused-attention kernel).

Caveats:
- The Blackwell-only `SM100f` triangle-attention kernel is not available on A100. The "FALLING BACK to the slow PyTorch path"
  warning in earlier smoke-test logs came from its tiny 18-residue protein, below cuEquivariance's 100-token cutoff
  (`CUEQ_TRIATTN_FALLBACK_THRESHOLD`, `CUEQ_TRIMUL_FALLBACK_THRESHOLD`). Mac1 (169 residues + ligand) is above it.
- cuEquivariance JIT-compiles a Triton helper. Polaris sets `CC=nvc`, which rejects `-Wno-psabi`, so
  `jobs/polaris/run_job.pbs` exports `CC=gcc` when `VENV_DIR` is set. Without it the first kernel call fails.
- The check confirms kernels launch in isolation; it does not profile the full Boltz run.

## Result (zero-shot, single seed)

| PDB | exp. IC50 (uM) | Boltz affinity (log10 uM) | P(binder) | iPTM |
|---|---|---|---|---|
| 9CY0 | 0.02 | 0.24 | 0.95 | 0.99 |
| 13RR | 1.9 | 2.20 | 0.51 | 0.97 |
| 7HC4 | 3.1 | 0.78 | 0.65 | 0.97 |
| 7HEC | 8.7 | 2.41 | 0.33 | 0.94 |
| 7HCF | 130 | 2.48 | 0.24 | 0.97 |
| 7HCO | >1000 | 0.96 | 0.45 | 0.97 |

All six structures ran without failures. iPTM is 0.94-0.99 for every ligand including the non-binder, so structure confidence
does not discriminate. Affinity ranks the most potent ligand (9CY0) first, but with six points and one seed this is anecdotal, not a benchmark.
Predicted structures have not yet been compared against the deposited ligand poses.

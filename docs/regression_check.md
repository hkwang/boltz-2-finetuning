# Regression check for fine-tuned Boltz-2

Co-folds three deposited protein-ligand complexes with stock `boltz2_conf.ckpt` and with a fine-tuned model, then scores both
against the crystal structures. It answers one question: did diffusion fine-tuning (Mac1) damage what the model already did
well elsewhere? It replaces the planned six-ligand Mac1 co-fold as the forgetting check (see `diffusion_finetuning_progress.md`, open item 1).

## Systems (`configs/regression.yaml`)

| Name | PDB | Chain | Ligand (CCD) | Resolution | Why |
|---|---|---|---|---|---|
| `streptavidin_biotin` | 3RY2 | A | BTN | 0.95 A | Tight, rigid, unrelated to Mac1 |
| `cdk2_staurosporine` | 1AQ1 | A | STU | 2.0 A | Kinase, ATP-site inhibitor |
| `abl_imatinib` | 1IEP | A | STI | 2.1 A | Kinase, DFG-out binding mode |

- One chain and one ligand per system. Sequences are read from the SEQRES of the downloaded CIF, so they match the crystal construct.
  Ligands are given as CCD codes (`ligand: ccd:`), so atom names match the deposited ligand and atoms are paired by name.
- `msa: empty` for every protein, as for Mac1: no MSA server call, and the input is identical before and after fine-tuning.
- All three were released before 2023-06-01 (Boltz-2's training cutoff), so Boltz-2 has very likely seen them.
  **This measures retained accuracy on familiar structures, not generalisation.** A fine-tune that drifts will show it here first;
  a pass says nothing about novel complexes.
- Settings: seed 2026, 3 recycling steps, 200 sampling steps, 5 diffusion samples (`model_0` is the top-ranked by confidence).
  These are inference settings, unrelated to the training `sampling_steps` in `configs/finetune_diffusion.yaml`.

## Metrics (`src/boltz2_finetuning/regression.py`)

Per sample, after pairing CA atoms and ligand atoms with the reference:

- `ca_rmsd`: CA RMSD over residues observed in both the crystal and the prediction (matched by SEQRES index), after superposing those CAs.
- `ligand_rmsd_global`: ligand heavy-atom RMSD after that same superposition.
- `ligand_rmsd_pocket`: ligand RMSD after superposing only CAs within 10 A of the crystal ligand. **This is the headline pose metric**;
  it is insensitive to domain motion elsewhere in the chain.
- A system is `ok` when `ligand_rmsd_pocket < success_rmsd` (2.0 A). `top_ok` uses `model_0`; `n_ok` counts the five samples.
- Also recorded: `confidence_score`, `iptm`, `ligand_iptm`.

Caveats: ligand atoms are paired by name with no symmetry correction, so a symmetric group flipped by 180 degrees (a phenyl or carboxylate)
inflates RMSD. Water, ions and the second chain of the crystal asymmetric unit are not modelled.

## Run

Needs the Boltz cache from `jobs/polaris/README.md` and, for the first run, network access to download CIFs
to `data/structures/regression/` (do that on a login node: `PYTHONPATH=src python experiments/regression_check.py prepare`).

```bash
# 1. Stock model (baseline)
TMPDIR=/tmp qsub -l select=1:ngpus=1 -l walltime=00:45:00 -N reg-baseline \
  -v JOB_SCRIPT="$PWD/experiments/run_regression.sh",VENV_DIR="$HOME/venvs/boltz2",TAG=baseline \
  jobs/polaris/run_job.pbs

# 2. Fine-tuned model: WEIGHTS is trainable_weights.pt written by experiments/train_diffusion.py
TMPDIR=/tmp qsub -l select=1:ngpus=1 -l walltime=00:45:00 -N reg-finetuned \
  -v JOB_SCRIPT="$PWD/experiments/run_regression.sh",VENV_DIR="$HOME/venvs/boltz2",TAG=<run name>,WEIGHTS=<path>/trainable_weights.pt \
  jobs/polaris/run_job.pbs

# 3. Side by side (login node)
PYTHONPATH=src python experiments/regression_check.py compare baseline <run name>
```

Each job takes about 2 minutes of `boltz predict` on one A100 (103 s for the baseline) plus start-up. Output goes to
`experiments/results/regression/<TAG>/` (`boltz_results_inputs/` with CIFs and confidence JSONs, `scores.csv`).

`WEIGHTS` works by `make-checkpoint`: it copies `boltz2_conf.ckpt` and replaces the tensors saved in `trainable_weights.pt`, giving an
ordinary Boltz-2 checkpoint passed to `boltz predict --checkpoint`. The diffusion-only fine-tune leaves every other tensor
unchanged (checked by hash in `train_diffusion.py`), and `make-checkpoint` fails if the names or shapes do not match
or if no tensor differs from the base. The checkpoint is a full copy (~2.2G), built in node-local `/tmp` and deleted when the job exits, because the home quota is small (earlier runs that wrote it into `experiments/results/` plus the perturbed weights hit `Disk quota exceeded`).

## Files

| File | Role |
|---|---|
| `configs/regression.yaml` | Systems, inference settings, success threshold |
| `src/boltz2_finetuning/regression.py` | RMSD/Kabsch scoring, reference extraction, fine-tuned checkpoint builder (not `evaluation.py`, which holds the Mac1 pose/affinity metrics) |
| `experiments/regression_check.py` | `prepare`, `perturb`, `make-checkpoint`, `score`, `compare` subcommands |
| `experiments/run_regression.sh` | PBS job body: prepare, optional checkpoint build, `boltz predict`, score |
| `testing/test_regression.py` | Local tests (no test for `perturb`; it is exercised by the positive-control jobs): scoring of rigidly moved and displaced copies, residue matching, checkpoint building |

Local check (login node, no GPU): `PYTHONPATH=src python testing/test_regression.py`.

## Result: stock model (job 7746657, 2026-10-15)

| System | CA RMSD (A) | Ligand RMSD, pocket (A), top | best of 5 | Ligand RMSD, global (A), top | iPTM | samples ok |
|---|---|---|---|---|---|---|
| `abl_imatinib` | 0.70 | 0.25 | 0.25 | 0.30 | 0.99 | 5/5 |
| `cdk2_staurosporine` | 0.51 | 0.32 | 0.30 | 0.30 | 0.99 | 5/5 |
| `streptavidin_biotin` | 1.03 | 0.81 | 0.79 | 0.83 | 0.97 | 5/5 |

All three ligands are reproduced within 1 A, so the baseline is near the ceiling; any fine-tune that moves a pose by more than
about 1 A is a clear regression, but this set cannot show improvement. Single seed, no variance estimate: run the baseline with
another `seed` in `configs/regression.yaml` to find the seed-to-seed spread before reading small differences.

## Result: 10-step timing fine-tune (job 7746664, tag `timing10`)

`WEIGHTS=experiments/results/finetune_diffusion/timing_1gpu/trainable_weights.pt` (10 steps at `max_lr` 1e-5, 779 tensors replaced).

| System | CA RMSD stock -> tuned (A) | Ligand RMSD, pocket, top: stock -> tuned (A) | samples ok |
|---|---|---|---|
| `abl_imatinib` | 0.70 -> 0.71 | 0.25 -> 0.26 | 5/5 -> 5/5 |
| `cdk2_staurosporine` | 0.51 -> 0.52 | 0.32 -> 0.32 | 5/5 -> 5/5 |
| `streptavidin_biotin` | 1.03 -> 1.04 | 0.81 -> 0.81 | 5/5 -> 5/5 |

No regression, as expected: ten steps at a tiny learning rate barely move the weights, so this run only shows that the pipeline works end to end
(the tuned checkpoint loads, differs from the stock one, and predicts). It is not evidence that a real fine-tune is safe. The 0.00-0.01 A differences are
below any seed-to-seed spread we have measured (none yet). We have not run a positive control, so we do not know that the check would flag a damaged model;
the cheapest control is to score a checkpoint whose diffusion weights were perturbed or trained at a large learning rate.

## Positive control: noisy diffusion weights (jobs 7746697-7746699)

Question: would this check flag a damaged model? `NOISE_SCALE=x` (or `regression_check.py perturb`) adds Gaussian noise with std `x` times each tensor's own std to every floating-point tensor of `structure_module` and
`diffusion_conditioning` in `boltz2_conf.ckpt` (839 tensors, seed 0; a real fine-tune saves 779 of these, the other 60 are `output_projection` weights/biases that
`train_diffusion.py` does not list in `trainable_weights.pt`, which I did not investigate), then runs the normal pipeline. Everything else is untouched.

Top-ranked sample (`model_0`), ligand RMSD after pocket superposition, and CA RMSD, in A; samples ok = of 5 within 2 A:

| System | stock | noise 0.1 | noise 0.3 | noise 1.0 |
|---|---|---|---|---|
| `abl_imatinib` CA | 0.70 | 3.81 | 21.26 | 24.52 |
| `abl_imatinib` ligand | 0.25 | 1.21 (5/5 ok) | 16.45 (0/5) | 23.28 (0/5) |
| `cdk2_staurosporine` CA | 0.51 | 2.95 | 21.73 | 25.55 |
| `cdk2_staurosporine` ligand | 0.32 | 0.50 (5/5 ok) | 14.02 (0/5) | 19.64 (0/5) |
| `streptavidin_biotin` CA | 1.03 | 3.30 | 15.32 | 21.34 |
| `streptavidin_biotin` ligand | 0.81 | 2.20 (0/5) | 13.90 (0/5) | 20.11 (0/5) |
| iPTM, mean of the three | 0.98 | 0.86 | 0.38 | 0.39 |

Reading it:

- **The check detects gross damage.** At 0.3 x and 1.0 x every ligand is 13-23 A off and 0/5 samples are within 2 A.
- **At 0.1 x the 2 A ligand threshold misses damage.** Two of three systems still pass, though CA RMSD rose from 0.5-1.0 to about 3-4 A and iPTM fell from 0.98 to 0.86. The protein fold moves
  before the ligand does. **Judge a fine-tune on `ca_rmsd_top` and `iptm_top` as well as the ligand pass flag**, and read the `compare` deltas rather than `top_ok` alone.
- **Rough tolerance.** The 10-step fine-tune moved CA RMSD by 0.01 A; 0.1 x noise moved it by 2.3-3.1 A. A real fine-tune lands somewhere between; we have no seed-to-seed spread yet,
  so call a change real only when it is well above 0.1-0.2 A in CA RMSD until that is measured.
- **Limits.** Random noise is not what training does. A fine-tune can degrade specific features (for example flipping a binding mode) without moving the fold, which this
  three-system set may not catch. It also cannot say whether a Mac1 fine-tune helps Mac1.

## Result: candidate fine-tunes (jobs 7749932, 7749933)

Tags `full_lr1e-4_step1400` (`finetune_diffusion/full_lr1e-4/weights/step1400.pt`, 779 tensors) and `lora_r16_lr3e-4_step2000`
(`finetune_diffusion/lora_r16_lr3e-4/weights/step2000.pt`, merged LoRA weights). Top-ranked sample; in A; stock -> tuned.

| System | CA RMSD, full | ligand pocket, full | CA RMSD, LoRA | ligand pocket, LoRA | iPTM stock / full / LoRA | samples ok |
|---|---|---|---|---|---|---|
| `abl_imatinib` | 0.70 -> 0.71 | 0.25 -> 0.20 | 0.70 -> 0.76 | 0.25 -> 0.20 | 0.99 / 0.98 / 0.99 | 5/5 all |
| `cdk2_staurosporine` | 0.51 -> 0.56 | 0.32 -> 0.25 | 0.51 -> 0.60 | 0.32 -> 0.27 | 0.99 / 0.99 / 0.99 | 5/5 all |
| `streptavidin_biotin` | 1.03 -> 1.05 | 0.81 -> 0.81 | 1.03 -> 1.04 | 0.81 -> 0.79 | 0.97 / 0.97 / 0.96 | 5/5 all |

**Both pass.** CA RMSD moves by +0.01 to +0.09 A (0.1 x noise moved it by 2.3-3.1 A), ligand pocket RMSD is unchanged or 0.02-0.07 A better,
iPTM moves by at most 0.01, and every sample is within 2 A. The changes are below the tolerance set above (0.1-0.2 A CA) and seed-to-seed spread is
still unmeasured, so read this as "no detectable regression", not as an improvement. LoRA moves CA slightly more than full fine-tuning on two systems.

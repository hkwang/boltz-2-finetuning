# Diffusion fine-tuning: progress log

Written 2026-10-14, updated 2026-10-15. Status: **1-GPU (job 7746606) and 4-GPU (job 7746627) 10-step timing runs both work**: finite losses, frozen weights bit-identical. Read this together with
[`boltz2_training_notes.md`](boltz2_training_notes.md) (what Boltz supports), [`data_schema.md`](data_schema.md) and [`datasets.md`](datasets.md).

## Decisions made (user)

- Train the **diffusion head** on all 557 crystal structures (`diffusion_train` = true in `data/processed/mac1_training_records.csv`), adapting the fork's `trainingv3.py`.
- **Do not modify the installed Boltz** (`~/venvs/boltz2`); it is the benchmark. All changes live in this repo.
- Affinity head, later: focal loss plus sample weights; one assay id per source (5 sources); racemic pairs use one entry matching the deposited conformation (first-listed code in `mac1_ligands.csv`); ignore the pre-Boltz-cutoff leakage for now.
- Zero-shot baseline to compare against: `experiments/results/mac1_cofold/summary.csv` (and the 4-GPU rerun). Six test-split ligands only; the notes suggest extending to all 25 test entries.

## Facts established

- **Censored IC50s:** 22 of 809 affinity rows are `>` (15 eLife, 7 Science Advances); 7 survive into the 182 usable pairs (2 train, 4 val, 1 test in the old split).
  Reason: HTRF peptide-displacement assay with no inhibition at the highest concentration tested. The raw column shows the ceiling: `> 1.0E+03` (1000 uM) for 18, `> 200` for 3.
  Value stored is the ceiling, so it is a lower bound on the IC50, not a measurement.
- **`trainingv3.py` (fork) does not work as is** for our data: it loads `pre_affinity_{id}.npz` (re-predicted poses), `yamls/{id}.yaml`, cached embeddings, forces `AffinityCropper` when affinity is on, and leaves `val_group_mapper` empty.
  Imports and featurizer call signature do match installed Boltz 2.2.1. The stock `boltz/data/module/trainingv2.py` is unusable (Boltz-1 featurizer).
- **Token counts of the 557 diffusion records:** min 181, median 359, 95th percentile 396, max 1103 (atoms median 2575, max 6793). They are 2-6 chains because of the crystal asymmetric unit, not just Mac1 plus ligand.
  A 512-token crop covers almost all records; the longest are cropped.
- **`Boltz2.training_step`** (installed, `boltz2.py:793`) works for diffusion only if `confidence_prediction=True`: `forward` builds `diffusion_conditioning` inside a branch gated on `(not self.training) or self.confidence_prediction`.
  So the confidence head is kept (frozen, loss weight 0) and costs compute. It also needs `batch size 1` and `feats["coords"]` with a conformer dimension.
- **No affinity loss exists in Boltz-2.2.1.** The affinity head needs its own loss and training loop (fork's `src/boltz/model/loss/affinity.py` is the reference; copies were in `/tmp/ext`, which is not persistent).
- **Checkpoint facts:** `boltz2_conf.ckpt` has `structure_prediction_training=False`, `affinity_prediction=False`, `ema=True` hyperparameter; `boltz2_aff.ckpt` has the affinity modules, `validate_structure=False`, and no stored optimizer state. Pretraining `max_lr` was 1e-3 with 1000 warmup steps; `diffusion_multiplicity` 32.

## Compatibility fixes needed to load `boltz2_conf.ckpt` into `Boltz2` for training

These are in `experiments/train_diffusion.py`; each came from a real failure:

1. `diffusion_process_args` contains `mse_rotational_alignment`, which the installed `AtomDiffusion` rejects -> drop it (jobs 7746197/7746198 died on this).
2. `pairformer_args` in the checkpoint lacks `v2: True` -> state_dict mismatch (`attention.norm_s.*` missing). `boltz predict` sets it from `PairformerArgsV2`; we add it (job 7746223).
3. `Boltz2.setup` reads `self.validate_structure`, which `__init__` never stores -> set `model.validate_structure = False` after loading (job 7746237).
4. `steering_args` must be a dict with all keys (the checkpoint has `None`) -> `STEERING_OFF` in the script.
5. `training_args` must be a `DictConfig` (the model uses attribute access) -> merged with `OmegaConf`.

Not yet checked: whether `use_kernels=True` plus bf16-mixed trains without error, whether `training_step` swallows exceptions and returns `None` (it does for the diffusion loss: "Skipping batch ... due to error"), so a run can look healthy while skipping every batch. **Check that the loss is finite and the step count advances.**

## Code written this session

| File | Purpose |
|---|---|
| `src/boltz2_finetuning/training_data.py` | `CrystalDataset` (load_input -> tokenize -> BoltzCropper -> `Boltz2Featurizer.process(training=True)`), `CrystalDataModule` (batch 1), `diffusion_record_ids`. Checked on the login node: collates a sample to correct tensors (357 tokens for one record). |
| `experiments/train_diffusion.py` | Lightning entry point: load checkpoint, freeze everything outside `trainable_modules`, hash frozen params before/after (must be bit-identical), per-step timing callback, `last.ckpt` resume, saves `trainable_weights.pt`. Accepts `key.sub=value` overrides. |
| `configs/finetune_diffusion.yaml` | Defaults: trainable `[structure_module, diffusion_conditioning]`, max_lr 1e-5, 20 warmup steps, diffusion_multiplicity 16, sampling_steps 20, recycling 3, distogram/confidence/bfactor loss weights 0, 10 steps (timing only). |
| `experiments/run_finetune_diffusion.sh` | PBS job body: kernel check, `nvidia-smi dmon` to `$DMON_LOG`, runs the trainer. |

Not yet done: unit test in `testing/`, README mention, a validation loop (none exists; `limit_val_batches=0`), the pretrain-regression check, and the affinity head.

## PBS jobs

| Job | What | Result |
|---|---|---|
| 7745696 / 7745741 | Zero-shot co-fold, 1 / 4 GPUs | finished, Exit 0 (earlier) |
| 7746197 (1 GPU), 7746198 (4 GPU) | First timing runs | failed at model construction (fix 1) |
| 7746223 | 1 GPU | failed at `load_state_dict` (fix 2) |
| 7746237 | 1 GPU | failed in `Trainer.fit` setup (fix 3) |
| **7746241** | 1 GPU, 10 steps, `name=timing_1gpu` | **submitted, result unknown** |

Output: `ft-diff-1gpu.o<jobid>` in the repo root; GPU log `experiments/results/finetune_diffusion/dmon_1gpu.log`; checkpoints and CSV logs under `experiments/results/finetune_diffusion/timing_1gpu/`.
The 4-GPU timing job has **not** been resubmitted; do that once 7746241 shows finite losses:

```bash
TMPDIR=/tmp qsub -l select=1:ngpus=4 -l walltime=00:40:00 -N ft-diff-4gpu \
  -v JOB_SCRIPT="$PWD/experiments/run_finetune_diffusion.sh",VENV_DIR="$HOME/venvs/boltz2",OVERRIDES="name=timing_4gpu trainer.devices=4 trainer.max_steps=10 data.num_workers=4",DMON_LOG="$PWD/experiments/results/finetune_diffusion/dmon_4gpu.log" \
  jobs/polaris/run_job.pbs
```

Model construction takes ~2-4 min on the loaded login node; on a compute node it is a smaller share, but expect minutes before step 1.

## Open items and risks

1. **Forgetting.** Only the diffusion module and its conditioning train; the trunk, distogram and confidence heads are frozen and hash-checked. That is still a real shift on 557 near-identical Mac1 structures (one protein, hundreds of copies). Regression check: `docs/regression_check.md` (streptavidin-biotin, CDK2-staurosporine, Abl-imatinib co-folded and scored against the crystal structures; run it on every fine-tuned `trainable_weights.pt`). The 10-step timing weights pass trivially.
2. **Repeated protein.** 557 records are one protein in many ligand states, so "epochs" overfit quickly. A tiny-LR, few-hundred-step schedule with a held-out structure RMSD is safer than long training.
3. **Starting checkpoint** is `boltz2_conf.ckpt` (the diffusion weights that `boltz predict` uses). `boltz2_aff.ckpt`'s structure weights were never compared with it.
4. **Hyperparameters are guesses** (lr 1e-5, multiplicity 16, sampling steps 20); only the timing run has been configured, no tuning.
5. **No validation signal** yet. Validation entries (21 val, 25 test) are excluded from diffusion training, so a pose-RMSD eval on them can be added by predicting with the fine-tuned weights (`trainable_weights.pt` holds only the unfrozen tensors; loading them into the stock model for `boltz predict` is not written).
6. **Known data caveats** carried from the datasets docs: 2 unmatched ligands (7HFU, 7HFX), unresolved AVI-4206/AVI-6451 IC50 mismatches in the J. Med. Chem. SI, the 4 stereoisomer-ambiguous entries, ~69 pairs probably seen in Boltz-2 pretraining.
7. `core.1271488` and old `*.o*` files sit in the repo root untracked; they can be deleted or moved by hand.

## Update 2026-10-15: first working training run

Job **7746606** (1 A100, `name=timing_1gpu`, 10 steps, Exit 0). Output `ft-diff-1gpu.o7746606`:

| Step | Record | Tokens | Loss | Seconds | Peak GPU |
|---|---|---|---|---|---|
| 1 | 5S3J | 350 | 1.038 | 19.7 (kernel JIT warmup) | 6.5 GB |
| 2-10 | various | 354-408 | 0.96-1.18 | 2.7-4.9 (median ~3.5) | 8.7 GB |

- Loss is stable around 1.0 (diffusion loss x 4 + 0; smooth-LDDT plus MSE); 10 steps at warmup LR cannot show a trend. GPU power ~150 W, SM ~40% (dmon): one record per step at 2.6k atoms underfills an A100.
- `frozen parameters bit-identical after training` printed. `trainable_weights.pt` vs `boltz2_conf.ckpt`: 640/779 trainable tensors changed, max abs difference 2.3e-5, as expected for 10 warmup steps at max_lr 1e-5.
- Rough cost: ~3.5 s/step at batch 1 and diffusion multiplicity 16 -> ~1 epoch of 557 records is ~33 min on one GPU, before the multi-minute model build (~2-4 min on a compute node).

### Further fixes after the list above (all in this repo)

6. `Boltz2.training_step` cannot run structure training with the confidence head on: `get_true_coordinates` asserts `coords.shape[0] == 1`, but `forward` has already expanded `coords` to `diffusion_multiplicity` copies. Fix: `src/boltz2_finetuning/model.py::Boltz2Diffusion`, a subclass with a diffusion + distogram-only `training_step` (no try/except, so bad batches fail loudly; the stock one silently returns `None`).
   Also overrides `on_train_epoch_end`, which reads confidence loggers that do not exist when `validate_structure=False`.
7. The checkpoint stores `structure_prediction_training=False`, which turns off the diffusion loss -> set True. The installed `compute_loss` rejects `add_bond_loss` -> dropped.
8. `symmetry_correction` is irrelevant now that `training_step` no longer calls `get_true_coordinates`.
9. A `Record` dataclass in the batch breaks Lightning's device transfer -> the data module returns `record_id` strings (`collate_with_ids`).
10. `sampling_steps` is only used for the confidence branch's structure sample during training, so the config uses 2 to keep that cost small.

### 4-GPU DDP

- Jobs 7746607 and 7746617 failed with "Expected to mark a variable ready only once" (reentrant `torch.utils.checkpoint` inside trainable modules under DDP).
- Job 7746620 turned off `checkpoint_diffusion_conditioning` and the score model's `activation_checkpointing`: DDP then ran steps 1-2 on all four ranks (loss ~1.0-1.09, 3.2 s/step),
  but **ran out of memory at step 2 (31-38 GB peak per GPU)**. The earlier note that memory is not a constraint was wrong: the 8.7 GB figure was measured with checkpointing on.
- **Job 7746627 (Exit 0): checkpointing back on plus `DDPStrategy(static_graph=True)`** (`trainer.ddp_static_graph` in the config). 4 ranks, 10 steps, peak 9.6-11.3 GB per GPU, loss 0.93-1.19,
  frozen parameters bit-identical. Steady-state ~2.8-4.9 s/step per rank (4 records per step, so ~1 epoch of 557 records is ~140 optimizer steps, ~10 min). `static_graph` assumes the same parameters
  are used every step; true here because the graph is fixed (no cropping-dependent branches seen in 40 rank-steps), but a crash with a "static graph" message on a later step would mean this assumption failed.
- Ten steps cannot show learning. No held-out metric exists yet, so the next real run has no way to say whether the fine-tune helps or hurts; that comes before any long run.

## Update 2026-10-15 (later): held-out evaluation, in progress

Goal (user request): a held-out metric for the diffusion fine-tune, covering **ligand pose RMSD and affinity**. A second agent owns the pretrain regression check (non-Mac1 / six-ligand baseline); this section is only the Mac1 held-out metric.
The stock-weights baseline has since been run (next section). No fine-tuned weights worth evaluating exist yet (the only `trainable_weights.pt` files are from 10-step timing runs).

### Held-out set

- 45 entries with a measured-or-censored IC50 and a deposited pose: 24 `test` (22 `=`, 2 `>`) and 21 `val` (18 `=`, 3 `>`). All are excluded from diffusion training (`diffusion_train` false).
- Every polymer chain in these 45 CIFs matches the 169-residue construct (>=150 residues matched per chain), so the same sequence as `configs/mac1_cofold.yaml` can be used.
- Ligand matching for these 45: 33 `stereo`, 12 `connectivity`; 1-3 candidate codes each. `affinity_code` in `mac1_training_records.csv` is the ligand to score.
- 12 of the 45 have table SMILES that differ from the SMILES re-derived from the deposited model after canonicalization: 14AB, 7HCF, 7HCJ, 7HCN, 7HCP, 7HDN, 7HDZ, 7HEC, 7HEW, 7HFC, 7HFD, 7HFE.
  I did not check whether each is an enantiomer or a tautomer. **Open decision:** which SMILES to give Boltz for these (the earlier zero-shot co-fold used the table SMILES). The pose metric below maps predicted atoms with the SMILES passed in, so it works with either, but chirality is matched (`useChirality=True`), so an enantiomer mismatch will raise instead of silently scoring.
- Unexplained: a quick join showed `smiles` in `mac1_ligands.csv` equal to `mac1_splits.csv` for only 89% of these entries. Possibly a join/duplicate-row artifact (I took the first row per `pdb_id`). Not investigated.

### Checkpoint finding (useful for the regression agent)

`boltz2_conf.ckpt` and `boltz2_aff.ckpt` have **bit-identical weights** for every shared structure-side module: `structure_module` (720 tensors), `diffusion_conditioning` (119), `pairformer_module` (3392), `msa_module` (226), `input_embedder` (89), `distogram_module` (2), `confidence_module` (453); max difference 0.0.
The conf checkpoint has 83 keys the aff one lacks; the aff one has 520 more (the affinity modules). So the starting-checkpoint question is moot for structure. All 779 trainable keys exist in both state dicts. A fine-tuned checkpoint for `boltz predict` can be made by overwriting those keys in either file.
Consequence: in a co-fold + affinity run, the fine-tuned structure module changes the coordinates that the unchanged affinity head reads, so affinity moves only through the pose.

### Code written (this session)

| File | Purpose |
|---|---|
| `src/boltz2_finetuning/evaluation.py` | `pose_metrics` (pocket-aligned, symmetry-aware heavy-atom ligand RMSD vs the deposited ligand; also pocket-CA and whole-chain CA RMSD), `read_prediction` (per-sample confidence), `pose_summary` and `affinity_summary` with bootstrap 95% CIs |
| `testing/test_evaluation.py` | Superposition and summary tests: Kabsch recovers a rigid motion, never reflects, symmetry-swapped atoms give zero RMSD, threshold fractions |

Metric definitions:
- **Pose:** superpose the predicted protein onto the deposited chain that has the most residues within 6 A of the ligand, using C-alpha atoms of those pocket residues; ligand RMSD is the minimum over RDKit atom-symmetry matches and over deposited ligand copies. Reported per sample: `rmsd_top1` (highest `confidence_score`) and `rmsd_best` (best of N samples), with median + bootstrap CI and fractions under 2 A and 1 A.
- **Affinity:** `affinity_pred_value` (log10 IC50 in uM) vs `log10_ic50_uM`. Spearman and Pearson (bootstrap CIs), RMSE/MAE/bias, on `=` rows only. `>` rows are scored one-sidedly (prediction at or above log10 of the ceiling counts as consistent) and reported separately.

### Status of the pose code

- Works on 9CY0 against the existing zero-shot prediction: ligand RMSD **0.50 A**, pocket CA 0.53 A, whole-chain CA 1.04 A (2 deposited copies, 2 symmetry matches).
- **Fails for the other five zero-shot ligands (13RR, 7HC4, 7HEC, 7HCF, 7HCO) with "no fully modelled deposited copy". This is a bug in my code, not missing data.** Cause found for 7HC4 and 13RR: the deposited ligand atoms are all in alternate locations with low occupancy
  (7HC4: altloc B occupancy 0.06 and altloc C 0.14; 13RR: altloc B occupancy 0.14), and the deposited residues include hydrogens (44 atoms in the file vs 26 heavy atoms expected). `first_altloc` in `evaluation.py` only accepts the blank or `A` altloc, so these copies are dropped.
  Fix to make: for each ligand residue, use the highest-occupancy altloc per atom name (or per residue) and skip hydrogens; then rerun the six cases. Occupancy of 0.06-0.14 also means those poses are weakly supported in the density, which should be reported alongside the RMSD (add an `occupancy` column).
  `testing/test_evaluation.py` does not cover the altloc path.
- Not yet verified after my last edits: `useChirality=True` in the symmetry match and the predicted-atom-order check were added after the unit tests last passed; the tests were re-run at the end of this session (see below).

### Next steps

1. Fix altloc/hydrogen handling in `ligand_copies`; add a test for it; rerun the six zero-shot cases; check all 45 held-out entries load (some may have no usable deposited copy).
2. Decide the SMILES policy for the 12 differing entries (above).
3. `experiments/prepare_mac1_eval.py`-style script: one Boltz YAML per held-out entry (protein from `configs/mac1_cofold.yaml`, `msa: empty`, affinity binder ligand), same predict settings as `configs/mac1_cofold.yaml` plus `diffusion_samples` > 1 so `rmsd_best` is meaningful.
4. Script that writes a fine-tuned `.ckpt` by overwriting the 779 trainable keys in `boltz2_conf.ckpt` and `boltz2_aff.ckpt` from `trainable_weights.pt`; check the diffusion state dict loads with `strict=True` through `boltz predict --checkpoint/--affinity_checkpoint`.
5. PBS job to predict the 45 entries with stock weights (baseline) and with the fine-tuned weights; `experiments/evaluate_mac1.py` to write a per-entry CSV and a summary JSON.
6. A longer fine-tuning run (the 10-step weights differ from the pretrained ones by at most 2.3e-5, so they will look identical to the baseline). Hyperparameters are still untuned guesses.

### Caveats for interpreting the numbers

- The held-out ligands bind the same pocket as hundreds of training structures, and the crystal ligand poses are not independent of the training set's chemistry series, so pose RMSD is optimistic. 45 entries, so confidence intervals will be wide.
- The pre-Boltz-2-cutoff leakage is being ignored per the user's instruction; it affects both stock and fine-tuned models and the baseline comparison only partly cancels it.
- `/tmp` is not persistent and has been wiped before (`/tmp/hkwang_scratch` and `/tmp/ext` disappeared); nothing needed is stored there now except the fork's source, which was not copied into the repo.

## Update 2026-10-15 (night): baseline, quota incident, move to /eagle

### Stock-weights baseline on the 45 held-out entries

Job 7746728 (4 GPUs, 6 min, 5 structure samples), output `experiments/results/mac1_eval/stock/` (`eval_summary.json`, `eval_per_entry.csv`). Scripts: `experiments/prepare_mac1_eval.py`, `experiments/run_mac1_eval.sh`, `experiments/evaluate_mac1.py`, `experiments/make_finetuned_checkpoint.py` (written, never run).
The `ligand_copies` altloc/hydrogen bug above is fixed: it uses the highest-occupancy altloc and skips hydrogens (`testing/test_evaluation.py`, 5 tests pass). Deposited ligand occupancy is low (median 0.34), so the pose reference is weak. SMILES come from the deposited model (`model_smiles`); the table SMILES mismatch only for 14AB.

| Metric | Value |
|---|---|
| Ligand RMSD, top1 | median 1.61 A (95% CI 1.43-1.93); 69% under 2 A; 18% under 1 A |
| Ligand RMSD, best of 5 | median 1.43 A; 76% under 2 A |
| Val / test median RMSD (top1) | 1.80 / 1.48 A |
| Affinity, 40 measured | Spearman 0.68 (CI 0.42-0.84); Pearson 0.70; RMSE 1.31 log10; bias +0.99 log10 |
| Affinity, 5 censored | 0 of 5 consistent |

### Quota incident

`/home` reached its quota (44.6 of 45 GB) and job 7746690 died with "Disk quota exceeded". Freed by deleting two regenerable 4 GB `last.ckpt` files from the timing runs and three ~6 GB core dumps from the failed OOM 4-GPU job (`core.*` in the repo root). Training checkpoints are about 4 GB each, so they must not be written to `/home`.

### /eagle layout

Root: `/eagle/GenesisHackathonOct26/hkwang/boltz2-finetuning/`.

| Path in the repo / home | Now |
|---|---|
| `experiments/results` | symlink to `$ROOT/experiments/results` (all run outputs and checkpoints) |
| `~/.cache/boltz` | symlink to `$ROOT/cache/boltz` (checkpoints, `mols/`, `mols.tar`) |
| `data/`, `~/venvs/boltz2` | left on `/home` (668 MB and 128 MB) |

Because the symlinks keep the old paths, no script had to change. Both copies were verified with `diff -rq` against the originals (2.3 GB results, 7.6 GB cache, 45,227 mol files). Job 7746798 (`testing/eagle_check.sh`) on a compute node read the checkpoints, `mols.tar` and a result file through the symlinks and measured 376 MB/s write and 538 MB/s read on `/eagle`. The jobs must request `-l filesystems=home:eagle`.
The original copies are still at `experiments/results.old` and `~/.cache/boltz.old` (about 10 GB). The shell rejected `rm -rf`, so delete them by hand: `rm -rf experiments/results.old ~/.cache/boltz.old`. They are untracked and unused.

### Next

1. Run a meaningful fine-tune with `output_dir` on `/eagle` (checkpoints are about 4 GB), then `experiments/make_finetuned_checkpoint.py` and `LABEL=finetuned CKPT_DIR=... experiments/run_mac1_eval.sh`, and compare with the baseline above.
2. Affinity head loss, hyperparameter tuning, unit tests for `training_data.py` and `model.py`.
3. Repo-root clutter: `*.o*` PBS output files, `gpu-monitor.*.csv`, `core.*`.

### Smoke test on /eagle and LoRA (job 7746841, passed)

`testing/eagle_smoke.sh` ran with every output on `/eagle`: a 6-step full fine-tune and a 6-step LoRA fine-tune on 4 GPUs, a LoRA resume from `last.ckpt` (6 to 8 steps), export to Boltz checkpoints (434 tensors replaced, 395 differ from pretrained), and a 2-entry co-fold plus scoring with `boltz predict` (97 s on 1 GPU). Frozen parameters stayed bit-identical.
New code: `src/boltz2_finetuning/lora.py` (own LoRA wrapper, no `peft`; `testing/test_lora.py` checks identity at init and that merged weights equal the adapted forward pass), `SaveWeights` callback and `lora:` config block in `experiments/train_diffusion.py`, `configs/finetune_diffusion_lora.yaml`, `experiments/run_finetune_sweep.sh` (train, then export and score `weights/step<N>.pt`). `configs/finetune_diffusion.yaml` is now the long-run recipe (4 GPUs, 1400 steps).
The resource plan and the three wave-1 jobs (7746856, 7746857, 7746858) are in [`finetuning_plan.md`](finetuning_plan.md).

### Wave 1 finished

All three 1400-step runs (full 1e-5, full 3e-6, LoRA r16 1e-4) finished with frozen weights bit-identical; median 3.45 s/step on 4 A100s (about 80 min of training). Held-out top-1 ligand RMSD median 1.61 A (stock) to 1.45 to 1.53 A; best-of-5 1.43 to about 1.15 A; paired top-1 changes are not distinguishable from zero; affinity unchanged. Full table, timing, an editing-a-running-script incident and wave 2 are in [`finetuning_plan.md`](finetuning_plan.md). Compare any evaluations with `python experiments/compare_evals.py`; monitor with `bash experiments/status_finetune.sh [run names]`.

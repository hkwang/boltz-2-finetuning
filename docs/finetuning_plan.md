# Fine-tuning plan and GPU budget

Written 2026-10-15. Budget: about 8 h on 3 Polaris nodes (12 A100-40GB; 96 GPU-hours), at most 4 GPUs per job. Runs are separate PBS jobs, one node each, so they run concurrently.

## Measured costs (4 GPUs, data parallel, 1 crop per GPU per step)

| Run | Trainable | s/step | Peak GPU memory | Checkpoint |
|---|---|---|---|---|
| Full (`structure_module` + `diffusion_conditioning`) | 280M | 3.5 | 9.6 GB | `last.ckpt` about 4 GB |
| LoRA (rank 16, alpha 32, all 434 linears in `structure_module.score_model`) | 10.5M | 4.7 | 5.4 GB | `last.ckpt` smaller (adapters + Adam state), still carries the frozen weights |

- 557 training crystal structures, 4 GPUs, one record per GPU per step: **about 140 steps per epoch**, about 8 min (full) to 11 min (LoRA).
- LoRA is slower per step here (extra small matmuls and no saving on the trunk forward pass, which dominates), but it needs half the memory and the exported weights are small deltas.
- Held-out evaluation (45 entries, 5 samples, 4 GPUs): about 6 min, plus about 2 min of startup. Smoke test with the LoRA checkpoint: 97 s for 2 entries on 1 GPU.
- Each job also pays about 3 min of model construction and a few minutes of checkpoint writes. `last.ckpt` and `weights/step<N>.pt` go to `/eagle` (set by `output_dir`).

## Wave 1 (submitted): learning rate and method comparison, about 2.5 h each, 3 nodes in parallel

All use 1400 steps (about 10 epochs; 1400 x 4 = 5600 crops), 50 warmup steps, constant learning rate, multiplicity 16, 2 sampling steps, seed 0, same data and split. Evaluate `weights/step{400,800,1400}.pt` on the 45 held-out entries.

| Job name | Method | max_lr | Why |
|---|---|---|---|
| `full_lr1e-5` | full (280M) | 1e-5 | the previous default; pretraining peak was 1e-3 |
| `full_lr3e-6` | full (280M) | 3e-6 | conservative, to see whether 1e-5 moves the pretrained model too far |
| `lora_r16_lr1e-4` | LoRA r16 | 1e-4 | adapters start at zero, so they tolerate a higher learning rate |

Per job: 1400 x 3.5 s = 82 min (full) or 110 min (LoRA), + 3 evaluations x 8 min + startup = about 2.2 h and about 2.7 h. Walltime requested: 3 h. **GPU-hours: about 3 x 4 x 2.5 = 30 of 96.**
Checkpoint cadence is 200 steps, so a killed job loses at most 200 steps (resubmit the same job; `train_diffusion.py` resumes from `last.ckpt`, tested in the smoke run for LoRA).

Outputs (all on `/eagle/GenesisHackathonOct26/hkwang/boltz2-finetuning/experiments/results/`, reachable through the repo symlink `experiments/results`):
- `finetune_diffusion/<name>/logs/` per-step CSV losses, `weights/step<N>.pt`, `boltz_ckpt/step<N>/` (pair of Boltz checkpoints for `boltz predict`);
- `mac1_eval/<name>_step<N>/eval_summary.json` and `eval_per_entry.csv`;
- baseline to beat: `mac1_eval/stock/` (ligand RMSD top1 median 1.61 A, 69% under 2 A; affinity Spearman 0.68, RMSE 1.31 log10).

## Wave 2 (about 5 h of the budget left), chosen after wave 1

Decide from the wave 1 evaluations and the loss curves. Candidates, in order of expected value:
1. Repeat the best configuration with 2 more seeds (the held-out set is only 45 entries, so one run cannot separate a 0.1 A difference).
2. Longer training (about 3000 steps, about 21 epochs) of the best method, if the held-out RMSD was still improving at step 1400.
3. LoRA rank or learning rate variations (rank 64, max_lr 3e-4) if LoRA is competitive with full.
4. Affinity head training, which needs a new loss (stock Boltz has none).
Reserve about 1 h at the end for evaluation and documentation.

## Caveats

- The metric is pocket-aligned ligand RMSD against low-occupancy deposited ligands (median occupancy 0.34), on 45 entries sharing a pocket and chemistry series with the training set. Treat differences under about 0.2 A as noise; bootstrap CIs are in `eval_summary.json`.
- "Better on the held-out set" is only about structures. A regression check on non-Mac1 targets is owned by another agent (`docs/regression_check.md`); do not adopt a fine-tuned model before it passes.
- Fine-tuning changes the structure module that the unchanged affinity head reads, so affinity can move only through the pose.
- Hyperparameters were never tuned before wave 1. The learning-rate values above are guesses bracketing the 10-step runs.

## Wave 1 results (jobs 7746856, 7746857, 7746858; all finished)

**Timing (4 A100s per job, measured over 5600 per-GPU steps each):** median 3.45 s per step for all three runs (10th to 90th percentile 2.2 to 4.6 s, set by crop size), so **1400 steps took about 80 min of training**; full and LoRA were the same speed here, unlike the 10-step smoke test (4.7 s for LoRA). Peak memory 11.3 GB (full) and 7.1 GB (LoRA). Each held-out evaluation (45 entries, 5 samples) took about 6 min of `boltz predict` plus startup, so a job with 3 evaluations ran about 1 h 45 min of walltime in total. Cost: 3 jobs x 4 GPUs x 1.75 h = **21 GPU-hours**.

| Weights | Top-1 ligand RMSD median (A) | <2 A | Best of 5 median (A) | Paired change in top-1 vs stock, median [95% CI] | Affinity Spearman |
|---|---|---|---|---|---|
| stock | 1.61 | 0.69 | 1.43 | | 0.68 |
| full 1e-5, step 1400 | 1.51 | 0.73 | 1.14 | -0.02 [-0.11, +0.01] | 0.67 |
| full 3e-6, step 1400 | 1.53 | 0.71 | 1.19 | -0.05 [-0.12, +0.00] | 0.66 |
| LoRA r16 1e-4, step 1400 | 1.45 | 0.73 | 1.17 | -0.02 [-0.05, +0.01] | 0.66 |

(All steps, from `python experiments/compare_evals.py`: step 400 and 800 are in the same range.)

Reading:
- Training loss falls (diffusion loss, mean of last 50 steps: 0.228 for full 1e-5, 0.237 for full 3e-6, 0.226 for LoRA, from 0.253 at the start), and **held-out ligand RMSD improves slightly and consistently**: top-1 median 1.61 to about 1.5 A, best-of-5 median 1.43 to about 1.15 A. But the paired per-entry change in top-1 RMSD is about -0.02 to -0.05 A with confidence intervals that include zero, so **no run is distinguishable from stock on top-1 RMSD**. The best-of-5 improvement is larger (0.2 to 0.3 A), which suggests the fine-tuned model samples better poses but the confidence score does not pick them out.
- **Learning rate, step count and method make no visible difference** (LoRA matches full fine-tuning at 4% of the trainable parameters). The learning rates tried (3e-6 to 1e-4) are probably all small relative to what changes the model; wave 2 raises them.
- **Affinity is unchanged** (Spearman 0.66 to 0.69, bias about +0.98 log10 for every run). Expected: the affinity head was not trained and reads the structure only through the pose.
- Sampling noise of the evaluation is not yet known; wave 2 measures it (`stock_seed1`, `stock_seed2`).

## Operational notes from wave 1

- **Do not edit a shell script that a running job will still call.** Bash reads scripts incrementally; I added `SEED` to `experiments/run_mac1_eval.sh` while 7746856/7746857 were evaluating, which corrupted a line ("finity: command not found", Exit_status 127). The step-1400 predictions had finished, so I ran `experiments/evaluate_mac1.py` on them by hand. The step 400 and 800 evaluations were not affected. Queued jobs read the script at start, so editing before they start is fine.
- The PBS `Exit_status` of a sweep job reflects its last command, so check for `SWEEP DONE <name>` in the output file.

## Wave 2 results (all jobs finished)

`python experiments/compare_evals.py` pairs each checkpoint with stock at the same prediction seed (45 held-out entries, top-1 pocket-aligned ligand RMSD in A).

| Weights | top-1 median | <2 A | best of 5 | paired change in top-1 vs same-seed stock |
|---|---|---|---|---|
| stock seed 0 / 1 / 2 | 1.61 / 1.86 / 1.84 | 0.69 / 0.62 / 0.62 | 1.43 / 1.38 / 1.30 | |
| full 1e-4, step 1400, seeds 0/1/2 | 1.44 / 1.41 / 1.80 | 0.76 / 0.78 / 0.69 | 1.03 / 0.90 / 1.08 | -0.03 / -0.10 / -0.04 |
| full 3e-4, step 1000 / 2000 | 1.53 / 1.63 | 0.73 / 0.71 | 0.99 / 1.06 | -0.06 / +0.01 |
| LoRA r16 3e-4, step 2000, seeds 0/1/2 | 1.46 / 1.39 / 1.56 | 0.67 / 0.80 / 0.71 | 1.25 / 0.97 / 1.05 | -0.02 / -0.09 / -0.02 |

- **Prediction-seed noise is 0.2 to 0.3 A on the top-1 median** (stock: 1.61, 1.86, 1.84 for the same weights). It is larger than any difference between my checkpoints, so only same-seed paired changes and best-of-5 are interpretable.
- Every fine-tune is slightly better than stock at the same seed (paired change -0.02 to -0.10 A, most CIs touching zero) and best-of-5 improves from 1.3-1.4 to 0.9-1.2 A. Learning rate (3e-6 to 3e-4), method (full vs LoRA) and step count are not separable. Full 3e-4 at 2000 steps is no better than 1e-4 at 1400, so larger rates stop helping.
- Affinity is unchanged (Spearman 0.66-0.70, bias +0.97 to +1.01), as the affinity head was not trained.
- Candidates for downstream use: `full_lr1e-4` step 1400 and `lora_r16_lr3e-4` step 2000. Neither is adopted until the pretrain regression check (other agent, `docs/regression_check.md`) passes.

## Affinity-head fine-tuning

**What trains.** `affinity_module1` and `affinity_module2` (8.7M parameters, 520 tensors of `boltz2_aff.ckpt`), starting from the pretrained weights. The trunk, structure module and confidence head are frozen, so the pose distribution is not touched.

**Poses.** `experiments/run_affinity_poses.sh` runs `boltz predict` with the `full_lr1e-4` step 1400 structure weights on every labelled complex, crystal positives included, so training and test rows go through the same pipeline. `experiments/cache_affinity_features.py` reruns the frozen trunk on the predicted pose (recycling 5) and stores the affinity module inputs (`s_inputs`, ligand rows and columns of `z`, representative coordinates, masks, molecular weight) as `.npz` under `/eagle/GenesisHackathonOct26/hkwang/boltz2-finetuning/affinity_cache` (training, 2644 files) and `affinity_cache_eval` (the 45 held-out entries). Cached values match `boltz predict` to within pose resampling (2.07 vs 2.01, 1.85 vs 1.76). One non-hit (`mac1_nonhit_1487`) has no pose and is skipped.

**Labels** (`experiments/prepare_affinity_inputs.py` to `data/processed/mac1_affinity/labels.csv`):

| Kind | Rows | Label |
|---|---|---|
| `ic50` | 124 (87 binary positive at IC50 <= 100 uM, 2 censored) | log10 IC50 in uM; assay id = source string |
| `hit` | 174 | binder, weight 1.0 |
| `nonhit` | 2346 | non-binder, weight 0.2 (`weak`) or 0.5 (`weak_htrf`) |

The 21 val and 24 test IC50 entries are held out and are the same 45 entries as the pose metric.

**Loss** (`src/boltz2_finetuning/affinity.py`, after `molecularinformatics/Boltz2_affinity`): Huber (delta 1) on log10 IC50, weight 0.5; pairwise-difference Huber within an assay id, weight 2.0; focal loss on binder vs non-binder (alpha 0.25, gamma 2), weight 1.0; per-sample weights. Censored rows contribute only to the binary term. The 100 uM threshold and the non-hit weights are my choices.

**Bugs fixed on the way.** Collate assigned into a copy of `z`; the non-kernel path ran out of memory at 39 GB, so `use_kernels=True`; the kernels' triangle-attention backward needs TF32, set in `AffinityHead.on_train_start` so every DDP rank gets it. Per-GPU batch 16 IC50 + 4 hit + 12 non-hit ran out of memory on 40 GB A100s (failed logs in `experiments/results/finetune_affinity/failed_oom`); the default is now 8 + 2 + 6 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Runs** (600 steps, 4 GPUs each, `experiments/run_train_affinity.sh`, held-out evaluation every 100 steps; stock baseline Spearman 0.68, RMSE 1.31, bias +0.99):

| Name | Change |
|---|---|
| `aff_lr1e-5` | max_lr 1e-5 (interactive node) |
| `aff_lr2e-5` | max_lr 2e-5 (job 7749270) |
| `aff_lr5e-5` | max_lr 5e-5 (job 7749271) |
| `aff_nopair` | max_lr 2e-5, pairwise weight 0 (job 7749272) |

## Affinity-head results (step 500)

All four jobs died just after step 500 of 600 on an NCCL all-reduce watchdog timeout (all three 4-GPU jobs at the same sequence number, 2602, within the same minute; not investigated). Step 500 weights (`weights/step500.pt`) and evaluations are complete, and step 500 is the step chosen. Numbers are on the 40 measured (`=`) held-out IC50s; paired deltas are 2000-resample bootstraps over entries. Poses for the fine-tuned heads come from the `full_lr1e-4` step 1400 structure weights (the cache in `affinity_cache_eval`); the stock-head rows are the `affinity_pred_value` columns of `experiments/results/mac1_eval/<run>/eval_per_entry.csv`.

| Pose | Head | Spearman | RMSE | Bias | MAE | Spearman val / test |
|---|---|---|---|---|---|---|
| stock (seeds 2026 / 1 / 2) | stock | 0.68 / 0.67 / 0.67 | 1.31 / 1.30 / 1.31 | +0.99 / +0.98 / +0.98 | 1.14 / 1.14 / 1.15 | 0.71 / 0.70 |
| `full_lr1e-4` s1400 (seeds 2026 / 1 / 2) | stock | 0.67 / 0.67 / 0.68 | 1.30 / 1.28 / 1.29 | +0.98 / +0.97 / +0.97 | 1.14 / 1.12 / 1.13 | 0.72 / 0.69 |
| `full_lr1e-4` s1400 | `aff_lr1e-5` | 0.81 | 0.99 | +0.65 | 0.82 | 0.86 / 0.79 |
| `full_lr1e-4` s1400 | `aff_lr2e-5` | 0.84 | 0.99 | +0.66 | 0.83 | 0.87 / 0.80 |
| `full_lr1e-4` s1400 | `aff_lr5e-5` | 0.84 | 0.95 | +0.60 | 0.81 | 0.84 / 0.87 |
| `full_lr1e-4` s1400 | `aff_nopair` | 0.81 | 0.95 | +0.61 | 0.79 | 0.83 / 0.79 |

Separating the pose effect from the head effect (paired, same 40 entries):

| Comparison | dSpearman [95% CI] | dMAE [95% CI] |
|---|---|---|
| Pose only: fine-tuned pose vs stock pose, stock head | -0.004 [-0.04, +0.02] | -0.004 [-0.03, +0.02] |
| Head only: `aff_lr1e-5` vs stock head, same fine-tuned pose | +0.14 [+0.02, +0.27] | -0.32 [-0.43, -0.20] |
| Head only: `aff_lr2e-5` | +0.16 [+0.03, +0.32] | -0.31 [-0.43, -0.18] |
| Head only: `aff_lr5e-5` | +0.17 [+0.01, +0.36] | -0.33 [-0.48, -0.17] |
| Head only: `aff_nopair` | +0.14 [+0.01, +0.28] | -0.35 [-0.48, -0.22] |

Reading:
- **The gain comes from the head, not the pose.** The structure fine-tune changes affinity predictions by a median of 0.04 log10 per entry and leaves Spearman and MAE unchanged; the trained heads improve both, with CIs excluding zero. Bias falls from +1.0 to about +0.6 log10 but is not removed.
- **Pose quality does not predict affinity error here.** Spearman between per-entry |error| and top-1 ligand RMSD is -0.12 (stock), +0.13 (fine-tuned pose, stock head), -0.08 (`aff_lr5e-5`); entries with a pose above 2 A are not worse (`aff_lr5e-5` MAE 0.76 for 10 such entries vs 0.82 for 30 under 2 A).
- The four heads are indistinguishable (CIs overlap widely). `aff_lr5e-5` is nominally best; the pairwise term makes no clear difference.
- **Censored entries:** 0/5 consistent for every model, stock and tuned.
- **Caveats.** Step and run were selected on these same 45 entries, so the gains are optimistic; there is no untouched test number. The tuned head has not been scored on stock-weight poses (no cache for them); given the zero pose effect above this is unlikely to matter, but it is not measured.

## Housekeeping

- `experiments/results.old` and `~/.cache/boltz.old` (about 10 GB) are left on `/home` after the move to `/eagle`; delete them by hand.
- PBS stdout files (`*.o*`) and `gpu-monitor.*.csv` now go to `logs/` (git-ignored); prune it once read.

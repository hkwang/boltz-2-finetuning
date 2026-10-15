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

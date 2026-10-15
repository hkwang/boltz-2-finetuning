# Reporting results

Every evaluation writes per-entry tables in one fixed layout, so results can be compared and plotted later (for example with `experiments/plot_results_glance.py`, which makes the at-a-glance figures in `docs/report.md`). Agents do not need to plot; they need to leave these files in place and log the run.

## Held-out set

- 45 entries in `data/processed/mac1_eval/eval_entries.csv` (val and test). Report val and test **pooled**, as in `docs/report.md`.
- Pose metrics use all 45 entries. Affinity metrics use the 40 entries with `ic50_qualifier == "="`.
- Never train on or select checkpoints with these entries.

## Seeds

Run every reported configuration at prediction seeds **2026, 1 and 2**. A single-seed result is preliminary; say so when you report it.

## Pose (diffusion fine-tuning)

Produce with `experiments/run_mac1_eval.sh` (`LABEL`, `CKPT_DIR`, `SEED`; `run_eval_seeds.sh` runs several at once), which calls `experiments/evaluate_mac1.py`.

- Path: `experiments/results/mac1_eval/<label>/eval_per_entry.csv` plus `eval_summary.json`.
- Label: `stock`, `full_lr<lr>_step<N>` or `lora_r<rank>_lr<lr>_step<N>`; seed 2026 has no suffix, other seeds append `_seed<N>` (for example `full_lr1e-4_step1400_seed1`).
- Required columns: `pdb_id`, `split`, `ic50_qualifier`, `log10_ic50_uM`, `rmsd_top1`, `rmsd_best`, `affinity_pred_value`.
- Headline metric: median `rmsd_best` (best-of-5 ligand RMSD, Å) over the 45 entries.

## Affinity (head fine-tuning)

Produce with `experiments/run_train_affinity.sh` (`experiments/train_affinity.py`); set `output_dir=experiments/results/finetune_affinity/<head>` in `OVERRIDES` and report heads at **step 500**.

- Heads on fine-tuned seed-2026 poses: `experiments/results/finetune_affinity/<head>/eval_step<N>.csv`.
- Heads on stock seed-2026 poses: `experiments/results/finetune_affinity/stockpose/<head>.csv`.
- Heads on seed-1 and seed-2 poses: `experiments/results/finetune_affinity/heads_on_poses/<pose_label>_seed<N>/<head>.csv`.
- Head names: `stock` (pretrained head) or `aff_<variant>` (for example `aff_lr1e-5`, `aff_nopair`).
- Required columns: `record_id` (`mac1_eval_<PDB>`), `split`, `log10_ic50_uM`, `pred` (predicted log10 IC50, µM).
- Headline metrics: Spearman ρ and RMSE (log10 µM) of `pred` against `log10_ic50_uM` over the 40 measured entries.

## Uncertainty

Report two kinds of 95% interval and always say which one you mean:

- **Over seeds:** t-interval of the three seed values (n = 3). This measures sampling reproducibility only.
- **Over entries:** bootstrap of the held-out entries (2000 resamples) for each seed. This is the dominant uncertainty with 40-45 entries.

## Logging a run

Add one line per run to `docs/diffusion_finetuning_progress.md` (or the relevant notes file) with the label, checkpoint, seeds, PBS job id and headline metrics. To add a run to the figures later, add its label to `POSE_RUNS` or `HEADS` in `experiments/plot_results_glance.py`.

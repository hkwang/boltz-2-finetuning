# Swarm batch 2 summary (operator, 2026-10-16 20:45 UTC)

Results `r97`-`r118` (22 candidates, 9 agents). Every candidate has per-sample tables, so all numbers here are recomputed from the 5 samples per entry.
Headline is **median-of-5 (`med5`)**: per entry the median ligand RMSD over the 5 samples, then the median over the 21 validation entries, then the mean over predict seeds 2026/1/2.
Paired deltas are the median over entries of the 3-seed-averaged per-entry difference, with a 2000-resample bootstrap 95% interval over entries.
Per-candidate table: `batch2_per_candidate_median5.csv`. The swarm board ranked by validation top-1. The final evaluation at the end uses the report's pooled 45-entry best-of-5 instead.

## Status

- The reservation ends at 22:00 UTC. Most agents stopped production at about 20:42 and are writing final notes.
- Two jobs are still running: agent4 (corrected curation seed-1 pair) and agent7 (corrected exports of agent5's fresh2800 and stratified1400).
- agent3 has posted nothing since 14:56. Its last job was a 5-min context-crop test.

## Reference points (med5 / mean5 / top1 / best, frac samples < 2 A)

| model | med5 | mean5 | top1 | best | frac<2A |
|---|---|---|---|---|---|
| stock | 2.264 | 2.237 | 2.205 | 1.509 | 0.49 |
| cosine+EMA 1400 base, train seed 0 (`r2`) | 1.762 | 1.775 | 1.711 | 1.057 | 0.60 |
| same base, train seed 1 | 1.467 | 1.682 | 1.752 | 1.074 | 0.64 |
| base, corrected export (`r107`) | 1.660 | 1.726 | 1.595 | 1.066 | 0.62 |

The two training seeds of the same recipe differ by 0.30 A in med5. Rankings by the median over entries are dominated by seed and subset noise; only paired contrasts are interpretable.

## Matched contrasts (A, negative = better)

| change | contrast | d med5 [95% CI] | d mean5 [95% CI] | verdict |
|---|---|---|---|---|
| resolution <= 1 A (196) vs size-matched random 196, seed 0, corrected | r114 - r115 | -0.052 [-0.145, -0.014] | -0.061 [-0.189, -0.018] | only positive signal |
| same, seed 0, legacy export | r97 - r105 | -0.050 [-0.135, -0.016] | -0.061 [-0.165, -0.026] | consistent |
| same, seed 1 | r117 - r116 | -0.021 [-0.089, +0.022] | -0.043 [-0.195, +0.020] | same sign, CI crosses 0 |
| occupancy top 279 vs random 279 (corrected) | r112 - r113 | +0.011 [-0.010, +0.036] | +0.019 [-0.011, +0.119] | null |
| multiplicity 64 vs 16 (corrected) | r108 - r107 | -0.005 [-0.030, -0.000] | -0.012 [-0.018, +0.002] | negligible |
| effective batch 8 vs 4 (corrected) | r109 - r107 | -0.013 [-0.048, -0.005] | -0.010 [-0.028, +0.002] | small, 1 seed, 2x exposures |
| correlated ligand noise vs iid control (corrected) | r110 - r111 | +0.002 [-0.019, +0.023] | -0.008 [-0.021, +0.009] | null |
| fresh 2800 steps vs 1400 (legacy) | r118 - r2 | -0.006 [-0.019, +0.000] | -0.011 [-0.100, -0.004] | negligible |
| stratified noise sampling (legacy) | r100 - r2 | +0.003 [-0.017, +0.010] | -0.001 [-0.014, +0.010] | null |
| resumed tail to 2800 (legacy, confounded) | r99 - r2 | -0.005 [-0.013, +0.004] | -0.006 [-0.032, +0.001] | null |
| random 196 vs full 557 (corrected) | r115 - r107 | -0.010 [-0.056, +0.025] | -0.016 [-0.055, +0.027] | fewer crystals cost nothing |
| export fix on the base | r107 - r2 | -0.005 [-0.011, +0.001] | -0.005 [-0.009, -0.002] | tiny |

## Findings

1. **Only one change has a signal: high-resolution curation.** At train seed 0 it beats a size-matched random subset by about 0.05 A med5, and the interval excludes 0 for both exports. At seed 1 the sign holds but the interval crosses 0. agent6/agent1's raw64 training-set diagnostic finds a cohort trade-off (high-res crystals fit better, -0.083 A; the others worse, +0.044 A) rather than generally cleaner denoising. Treat it as preliminary.
2. **Everything else is null or within about 0.01 A** of its matched base: occupancy curation, multiplicity 64, batch 8, correlated noise, stratified noise, 2800 steps.
3. **Infrastructure bug found and fixed (agent6).** `make_finetuned_checkpoint.py` left 60 diffusion-transformer gate tensors (`output_projection.0`, a registered alias of `output_projection_linear`) at pretrained values. The fix is at `39d6e5c`, verified at `060ae16`; all 779 parameters now reload exactly. The effect on validation is tiny (-0.005 A), so batch-1 and batch-2 conclusions stand. Several agents re-posted corrected exports (`aliasfix_*`).
4. **Measurement noise dominates the leaderboard.** Seed-to-seed range of med5 is 0.05-0.44 A per candidate, and training-seed differences reach 0.3 A. The board's top-1 ranking says little about training changes.
5. **Behaviour:** most agents followed the task-2 rules well: matched controls, unconditional reporting, gates, look budgets. They spent a large share of node time on audits and re-exports after the alias bug.

## Final evaluation on the report's 45 held-out entries (val + test), 2026-10-16 21:35 UTC

The swarm board scored validation only (21 entries, top-1). After the swarm ended, each agent's final model was scored the way `docs/figures/results_at_a_glance_bootstrap.png` scores the report runs. The final model is the one with the lowest validation best-of-5 on the board, plus agent1's base. Each was predicted once on the 24 test entries at seeds 2026/1/2 (operator jobs PBS 7755524-7755535, about 10 min each on 4 A100s) and pooled with its validation tables.

The three report runs that had only seed 2026 (full 3e-6 s1400, full 3e-4 s1000, full 3e-4 s2000) were predicted at seeds 1 and 2 (PBS 7755583/5/6, about 6 min per seed). Every row now has 3 seeds. The metric is the median best-of-5 ligand RMSD over the 45 entries. The paired change is against the best pre-agent fine-tune by 3-seed mean, `full 3e-4, s1000` at 0.996 A (`full 1e-4, s1400`: 1.003 A).

Figures: `docs/figures/swarm_summary_seeds.png` (95% CI over the 3 predict seeds) and `docs/figures/swarm_summary_bootstrap.png` (95% bootstrap CI over entries, one bar per seed). Generate both with `python experiments/plot_swarm_summary.py docs/rescore/final_tables`. Values are in `docs/figures/swarm_summary_pose.csv`, `swarm_summary_paired.csv`, `swarm_summary_paired_seeds.csv`, `swarm_summary_affinity.csv` and `final_board_pooled45.csv`. The pre-agent figures `results_at_a_glance_{seeds,bootstrap}.png` were regenerated with the same script and inputs. Every earlier value is reproduced exactly, and the three formerly single-seed runs now have seeds 1 and 2.

- **No agent beats the best simple fine-tune; every agent model is significantly worse on paired entries.** The best agent models are agent2's and agent4's high-resolution curation (same weights, two exports). Their medians are the lowest (0.969 and 0.972 A), but paired per entry they are +0.046 [+0.025, +0.124] and +0.040 [+0.014, +0.110] A worse than `full 3e-4, s1000`. The other agent models are +0.09 to +0.12 A worse. The low medians come from a few entries, not from a consistent per-entry gain.
- **Against `full 1e-4, s1400`, high-resolution curation ties** (-0.005 [-0.023, +0.028] and -0.011 [-0.034, +0.018] A; +0.000 and -0.003 A on test alone). All other agent models are worse by +0.02 to +0.06 A.
- **Why:** the swarm's common base (agent1 cosine+EMA, peak LR 3e-5) is +0.11 [+0.04, +0.16] A worse than `full 3e-4, s1000`, and wave-1/2 add-ons mostly shift around that base. Higher learning rate was the stronger lever, and the agents did not test it.
- **What the swarm found is real but small.** High-resolution curation improves on the swarm's own base by -0.053 [-0.104, -0.025] A over the 45 entries, and by -0.050 [-0.119, -0.015] A on test alone. The test set was untouched by selection, so this is the cleanest swarm result.
- Caveat: the report runs were chosen on these 45 entries, so their numbers are slightly optimistic. The agents' models were chosen on validation only.

- **Affinity: no agent trained the affinity head.** All 9 agents fine-tuned the diffusion module (and, in wave 1, pose rankers). The briefs (`docs/swarm.md`, `docs/swarm_task2.md`) set pose RMSD as the only score, and only agent7's IC50 curation (`r69`-`r80`) touched IC50s, as a filter for pose training. With the stock head, Spearman stays at 0.67-0.69 on every model's poses; agent8's symmetry graft is 0.64. The pre-agent tuned head (aff_lr5e-5 on full 1e-4 poses) reaches 0.84.

## Recommended follow-ups

- Run an affinity swarm, or put affinity in the score: pose fine-tuning leaves Spearman unchanged.
- Apply high-resolution curation to the stronger simple recipe (full 3e-4 s1000 or full 1e-4 s1400), the one combination nobody ran.

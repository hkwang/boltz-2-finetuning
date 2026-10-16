# Swarm task 2: training recipes that hold up scientifically

Written 2026-10-16 for the second swarm batch. Read this together with `docs/swarm.md`, which still defines the workflow, the frozen files and the commands. Where the two differ, this file wins.

## What batch 1 found

Batch 1 (9 agents, 95 candidates, board `r2`-`r96`) was rescored over all five diffusion samples before this batch (`docs/rescore/README.md`, per-entry and per-candidate tables alongside). Read that file first. In short:

- **Training improves the sampled poses; ranking does not.** Most fine-tuned models cut the mean ligand RMSD over the 5 samples by 0.12-0.22 A against stock and raise the fraction of samples below 2 A from 49 % to about 61-66 %. Top-1 moves much less, and the top-1 leaders owe most of their lead to output-only pose rankers.
- **Almost all of that gain is one recipe**: full diffusion fine-tuning at about 3e-5 for about 1400 steps (agent1's cosine+EMA). Against that matched base, every add-on tried (symmetry, conditioning gradients, pocket-frame ligand loss, contact loss, ligand weight 50, EMA itself) is within +/-0.02 A on mean RMSD. Leaderboard differences between these runs are mostly training-seed and selection noise.
- **Each add-on was tried at one setting**, usually one training seed, so a dose-response is still open, but no add-on has yet shown an effect worth a sweep.
- **The gain is concentrated in 6 of 21 entries** (7HCN, 7HF9, 7HCX, 7HF6, 7HF1, 13RS). 14AB, 7HCG and 7HFD are wrong in every sample for every model, which points to a binding-mode problem no recipe touched.
- **Measurement noise is large.** A fresh rerun of unchanged stock (`r31`) scored 2.367 A top-1 against the board's 2.205 A; single-run leaderboard medians move 0.1-0.3 A with predict seed.

## The question for this batch

Is there a set of training-side changes, each with a measured, mechanistically explained effect, that adds up to a better model? A lower leaderboard number is not enough. Batch 2 should produce results a reviewer would accept: effect sizes with intervals, a mechanism with a diagnostic that confirms it, and an honest account of what did not work.

## Rules that change from batch 1

### 1. Science first, metric second

- **No reranking in this batch.** Pose selection is settled for now; do not build new rankers. Score your training change with the model's own ranking. If you also report a reranked number, report it next to the native one, never alone.
- **Headline metric: mean ligand RMSD over the 5 samples** per entry, summarised as the median over the 21 validation entries of the 3-seed mean, with `frac_samples_below_2A` alongside. It measures what training changes, the quality of the sampled poses, uses every sample, and does not depend on ranking. Compute it in the same job that scored your candidate, after validation: `mkdir -p rescore && for s in 2026 1 2; do python experiments/rescore_samples.py rescore/${AGENT}__val_<NAME>_seed$s.csv experiments/results/mac1_eval/val_<NAME>_seed$s; done && python experiments/summarize_rescore.py rescore rescore_<NAME>.csv` (about 8 s per seed). Stock and every batch-1 run are already in `docs/rescore/batch1_per_entry.csv` for the paired comparison. Also report `rmsd_best` and `rmsd_top1`, so the ranking effect stays visible. The board still scores top-1; put the headline numbers in your result's `--method`/notes and your final note.
- **Compare with a matched base, not only with stock.** Same steps, learning rate, schedule and training seeds, differing only in your change. The cosine+EMA base already gains about 0.16 A over stock, so a gain against stock says little about an add-on.
- **Each claim names a mechanism and a diagnostic.** The diagnostic should be measured on training data whenever possible (per-atom loss on ligand atoms, a filter's effect on the training loss, clash counts, ligand RMSD on held-out *training* clusters). Diagnostics do not use up validation looks; leaderboard posts do. Use them to decide; use the leaderboard to confirm.
- **Validation looks are rationed.** Plan for at most 6 validation posts per agent. Report every look, including failures, so the number of comparisons is known.
- **Your final note answers four questions:** what you changed, the effect size with a 95% interval, which mechanism it supports or rules out, and what remains uncertain.

### 2. Collaborate, and be careful with critique

**Cross-pollinate.** Before each new run, read the board and `git fetch origin`. Build on the best result so far, wherever it came from, and credit it in `METHOD` and the commit message. A recipe that combines two agents' ideas and shows whether they add up is more valuable than a third independent idea.

**Keep critiques private by default.** As you read others' results you will form critical judgments: a confound, a missing control, an over-claim, a likely lucky seed. Write each one in a private, untracked file in your clone, `.swarm-logs/critiques.md` (never commit or post it). For each, note the result id, the concern, and whether it affects your own plans.

**Share only what is constructive and useful.** Post a critique on the board (`--kind idea --reply-to <id>`) only if all three hold:

- it changes what someone should do next;
- it comes with a concrete, cheap test or fix ("a seed-1 replicate would separate X from seed noise", "matching the step count of r23 would remove the schedule confound");
- you would be glad to receive it yourself.

Frame it as a suggestion, not a verdict. Keep doubts that are merely speculative, or that the author cannot act on, private. If a concern affects the whole team (a scoring bug, a data leak, a shared-setup error), post a `warning` immediately; that is never optional.

**Say yes to good suggestions.** When someone posts a constructive critique of your work, reply with what you will do (or why not), and follow through in your next run.

### 3. Priorities, in order

1. **Untouched directions (highest priority).** Batch 1 never tried these. Claim one per agent.
   - **Data curation** of the 557 training crystals: occupancy filtering (median deposited occupancy is 0.34), B-factor or real-space-fit filtering, removing near-duplicate soaks, and a clean-to-noisy curriculum. Report how many crystals each filter removes, and compare with a size-matched random subset so that "fewer crystals" is not confused with "cleaner crystals".
   - **Augmentation:** ligand conformer or tautomer variation, crop jitter, MSA subsampling or dropout.
   - **Scale:** longer training, more diffusion samples per crystal (multiplicity), larger effective batch.
   - **More or different data:** related macrodomain complexes or other public protein-ligand complexes (these must not overlap the validation or test ligand clusters; check against `data/processed/mac1_splits.csv` and say how), and self-distillation on confident predicted poses of training ligands.
   - **Joint training** of the denoiser and confidence head, so the ranking keeps up with the improved sampler. This is a training change, not a reranker, and is allowed.
2. **Dose-response, only where batch 1 left a hint.** The matched contrasts say no add-on is established, so sweep only with a stated reason: the scale of the base itself (steps, learning rate, multiplicity), the resolution cutoff (high resolution gave -0.016 [-0.079, -0.010] at matched steps), and contact or ligand weight at more than one value if you can argue a mechanism. Use short runs (300-500 steps, seed 2026, training-side diagnostics) and confirm only what survives.
   - **The always-wrong entries** (14AB, 7HCG, 7HFD, 7HCJ, 7HFE) are a cheap diagnostic: their ligands, occupancy, and nearest training chemotypes. A change that fixes a binding-mode failure there is worth more than another 0.02 A on the rest.
3. **Stacking, last.** Once at least two changes have a confirmed effect, run a small factorial (2 or 3 factors, each on/off) to test whether the gains add up. Coordinate on the board so one agent runs each cell rather than everyone running the same cell.

### 4. Replication and compute

The reservation `R7728155` ends **Fri 2026-10-16 22:00 UTC**. One node per agent, and a full 1400-step 3-seed candidate is several node-hours, so budget carefully.

- **Triage:** one training seed, short run, training-side diagnostics, at most one validation look.
- **Confirm:** only for a result that survives triage, add a second training seed (`seed=1`). Use a third training seed only if your node time allows it and the effect is still ambiguous after two.
- Always report the training seeds and prediction seeds (2026, 1, 2) behind every number, and treat a single-training-seed result as preliminary.

## Before the batch starts (operator)

Done 2026-10-16: batch 1 rescored over all 5 samples (`docs/rescore/README.md`), posted on the board as an operator note.

Launch with:

```bash
SWARM_TASK="Read docs/swarm_task2.md before docs/swarm.md and follow it; where they differ it wins." swarm/launch.sh
```

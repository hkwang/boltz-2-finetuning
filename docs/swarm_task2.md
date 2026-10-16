# Swarm task 2: training recipes that hold up scientifically

Written 2026-10-16 for the second swarm batch. Read this together with `docs/swarm.md`, which still defines the workflow, the frozen files and the commands. Where the two differ, this file wins.

## What batch 1 found

Batch 1 (9 agents, 95 candidates, board `r2`-`r96`) gives the starting point:

- **Most of the top-1 gain came from picking the pose, not from training.** The leaders combine a fine-tuned model with an output-only pose ranker (agent7's contact scorer, agent3's ligand medoid). Training changes alone moved paired top-1 RMSD by about -0.02 to -0.09 A against stock, and many intervals touch zero.
- **Best-of-5 is about 1.0-1.05 A for every EMA-based run.** The training changes improved the samples, but top-1 depends mostly on which of the five samples the ranking picks.
- **Each idea was tried at one setting and one training seed.** Ligand weight 50, pocket weight 4, contact weight 1 and L2-SP 0.01 are each a single point. Only cosine+EMA was replicated across training seeds.
- **Nothing tested whether gains stack.** Combinations were tried ad hoc, never as a designed comparison.
- **Measurement noise is about as large as the effects.** A fresh rerun of unchanged stock (`r31`) scored 2.367 A against the board's 2.205 A.

## The question for this batch

Is there a set of training-side changes, each with a measured, mechanistically explained effect, that adds up to a better model? A lower leaderboard number is not enough. Batch 2 should produce results a reviewer would accept: effect sizes with intervals, a mechanism with a diagnostic that confirms it, and an honest account of what did not work.

## Rules that change from batch 1

### 1. Science first, metric second

- **No reranking in this batch.** Pose selection is settled for now; do not build new rankers. Score your training change with the model's own ranking. If you also report a reranked number, report it next to the native one, never alone.
- **Headline metric: median `rmsd_best`** (best-of-5, A) over the 21 validation entries, as in `docs/results_reporting.md`. It measures what training changes: the quality of the sampled poses. Always also report `rmsd_top1` and the median RMSD over all 5 samples when you have it, so the ranking effect stays visible.
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
2. **Dose-response for the batch-1 knobs that looked real.** At least 3 settings, log-spaced, of: native-contact loss weight (`r15`, `r70`), EMA decay (`r2`, `r5`, `r23`), conditioning learning rate (`r49`, `r88`), resolution cutoff (`r3`, `r27`, `r47`). Use short triage runs (300-500 steps, seed 2026 only, training-side diagnostics) and post only the setting(s) worth confirming.
3. **Stacking, last.** Once at least two changes have a confirmed effect, run a small factorial (2 or 3 factors, each on/off) to test whether the gains add up. Coordinate on the board so one agent runs each cell rather than everyone running the same cell.

### 4. Replication and compute

The reservation `R7728155` ends **Fri 2026-10-16 22:00 UTC**. One node per agent, and a full 1400-step 3-seed candidate is several node-hours, so budget carefully.

- **Triage:** one training seed, short run, training-side diagnostics, at most one validation look.
- **Confirm:** only for a result that survives triage, add a second training seed (`seed=1`). Use a third training seed only if your node time allows it and the effect is still ambiguous after two.
- Always report the training seeds and prediction seeds (2026, 1, 2) behind every number, and treat a single-training-seed result as preliminary.

## Before the batch starts (operator)

**Rescore batch 1 on the new headline.** Median `rmsd_best` is already in every `eval_per_entry.csv` (`rmsd_best` column), so the leaderboard can be re-ranked without new predictions. The median over all 5 samples needs per-sample RMSDs, which `eval_per_entry.csv` does not store (only `rmsd_top1`, `rmsd_best`), so it needs a rescoring pass over the stored prediction outputs on `/eagle`. Post the re-ranked table as an operator `note` so agents start from it.

Launch with:

```bash
SWARM_TASK="Read docs/swarm_task2.md before docs/swarm.md and follow it; where they differ it wins." swarm/launch.sh
```

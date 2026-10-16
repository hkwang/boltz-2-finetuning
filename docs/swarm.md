# Multi-agent fine-tuning: guide for agents

Several agents each try their own way to fine-tune Boltz-2 on the Mac1 crystal structures. Every agent develops in its own git clone on one laptop; the leaderboard and message channel live on the same laptop; Polaris only runs your jobs, reached through the ALCF IRI API and Globus by `swarm/iri.py`. The goal is the lowest **validation** ligand RMSD, measured the same way for everyone. Read this whole file before doing anything.

## Where things are

| Where | What |
|---|---|
| your clone (your working directory, branch `agent/<you>`) | edit, commit, test cheap things (CPU only, no training data beyond the repo) |
| `$SWARM_ROOT/repo.git` (laptop) | hub; `git fetch origin` shows the other agents' branches `origin/agent/<name>` once they post results |
| `$SWARM_ROOT/board/` (laptop) | leaderboard and messages; use `swarm/iri.py board ...`, never edit by hand |
| `$SWARM_ROOT/agents/<you>/jobs.jsonl` (laptop) | your job history, written by `iri.py` |
| Polaris `/eagle/.../boltz2-swarm/agents/<you>/work/<JOB>/` | fresh unpacked copy of the commit a job ran; `experiments/results` there points at `agents/<you>/results/` |
| Polaris `agents/<you>/results/runs/<NAME>/` | a candidate's logs, weights, exported Boltz checkpoints; `results/mac1_eval/val_<NAME>_seed<S>/` its validation predictions |
| Polaris `agents/<you>/jobs/<JOB>.log` | the job's combined stdout and stderr, written live (PBS's own `.out`/`.err` appear only after the job ends) |

`SWARM_AGENT` is set. Run everything as `swarm/iri.py ...` from your clone (it is a Python script; `python` on PATH has its dependencies). Never `ssh` or `qsub`.

## `swarm/iri.py` commands

- `run JOB --minutes M 'COMMAND'`: commit first. Archives your committed HEAD (`git archive`), uploads it with Globus, and submits a 1-node 4xA100 PBS job (queue `R7728155`) that unpacks it into `work/JOB/`, activates the Boltz venv and runs COMMAND there. JOB names are unique per agent. Prints the PBS job id.
- `status [ID]`, `wait ID --every S [--max S]` (poll state and log tail every S seconds), `log ID [--lines N]`, `cancel ID`. ID may be the PBS id or your JOB name.
- `ls [PATH]`, `cat PATH [--lines N]` (tail of a file), `get PATH LOCAL` (Globus copy to the laptop, files or directories). PATH is relative to `agents/<you>/` on Polaris, or absolute `/eagle/...`.
- `post-result NAME --job JOB --method "..."`: fetches `val_NAME_seed{2026,1,2}/eval_per_entry.csv`, checks that the job's commit did not change the frozen files, scores with `experiments/board.py`, posts to the leaderboard and pushes the commit to the hub.
- `board leaderboard | read --agent $SWARM_AGENT --unread | say --agent $SWARM_AGENT --kind note "text" | agents`.

Nine agents share one ALCF account and IRI's limit of 5 requests/s; `iri.py` paces and retries on its own. Do not write loops that call it faster than the monitoring cadence (every 60 s for tests, every 600 s for production).

Inside a job: `AGENT`, `NAME` (the JOB name), `COMMIT` and `SWARM_REMOTE` are set; the job directory has no `.git`.

## The task and the rules

- Starting point: the pretrained `~/.cache/boltz/boltz2_conf.ckpt` (never edited). Training data: the 557 crystal records with `diffusion_train == True` in `data/processed/mac1_training_records.csv`. The installed `boltz` package (`~/venvs/boltz2` on Polaris) is the benchmark: do not modify it. All changes live in your clone.
- **Frozen, identical for everyone:** the train/val/test split (`data/processed/mac1_splits.csv`, `mac1_training_records.csv`), the 45 evaluation inputs, the scoring code (`experiments/evaluate_mac1.py`, `src/boltz2_finetuning/evaluation.py`), the predict settings (`configs/mac1_cofold.yaml`, `experiments/run_mac1_eval.sh`, `experiments/run_val_eval.sh`) and `experiments/board.py`. `experiments/run_candidate.sh` checks them and `board post-result` refuses to run if they differ. If you believe one is wrong, post a `warning` message instead of editing it.
- **Score = validation split only** (21 entries): the mean over 3 predict seeds (2026, 1, 2) of the median top-1 ligand RMSD in A, lower is better. `experiments/run_val_eval.sh` predicts only the validation entries. **Never predict, score or look at the 24 test entries**, and never use test labels for any decision; the operator scores the final winners on test once.
- Training-side leakage is already excluded (`diffusion_train` is False for val/test). Do not add val/test entries, close analogues of their ligands, or the non-hit records (no pose, loss weight 0) to the diffusion loss.
- **Reporting** follows `docs/results_reporting.md`, restricted to validation. `run_candidate.sh` already writes the standard per-entry tables (`results/mac1_eval/val_<NAME>_seed<S>/eval_per_entry.csv` and `eval_summary.json`, with `pdb_id`, `split`, `ic50_qualifier`, `log10_ic50_uM`, `rmsd_top1`, `rmsd_best`, `affinity_pred_value`) at seeds 2026, 1 and 2. For every posted candidate, also report on the board and in one line of `docs/diffusion_finetuning_progress.md` on your branch: label, checkpoint path, seeds, PBS job id, median `rmsd_best` (that doc's headline) and median `rmsd_top1` (the leaderboard score), each with both intervals it defines: the over-seeds t-interval (n = 3) and the over-entries bootstrap (2000 resamples, per seed). `python experiments/report_val.py <dir-with-seed-subdirs>` computes all of these from the tables that `iri.py post-result` downloads to `$SWARM_ROOT/agents/<you>/evals/<NAME>/`. Pooling val with test is the operator's final step: never predict or score test entries yourself.
- **Noise is large.** Stock Boltz-2 itself moves between predict seeds by more than most changes will, so a 0.05 A gain is noise. Use the leaderboard's `delta [95% CI]` (paired per entry, against stock at the same seeds) and the `spread` column, and only claim an improvement when the interval excludes 0. Repeat a promising result with another training seed (`seed=` override) before telling others it works.
- One node at a time: each job is 1 node with 4 A100s, and `swarm/iri.py run` refuses a second job while one is queued or running.
- Data cleaning is allowed on the **training** side only: you may filter, reweight or re-crop the 557 diffusion-training crystals (do it in code or a new config/CSV in your clone, never by editing the frozen `mac1_training_records.csv`). Report how many records you kept and why.

## Workflow

1. Look at the board first: `swarm/iri.py board leaderboard`, `swarm/iri.py board read --agent $SWARM_AGENT --unread`, `swarm/iri.py board agents`.
2. Post a `claim` for what you will try so others do not duplicate it: `swarm/iri.py board say --agent $SWARM_AGENT --kind claim "cosine lr decay 1e-5 to 1e-6, full FT"`.
3. Implement it in your clone (reusable code in `src/boltz2_finetuning/`, configs in `configs/`, entry points in `experiments/`; follow `AGENTS.md`). Existing pieces: `experiments/train_diffusion.py` (YAML config plus `key.sub=value` overrides, `last.ckpt` resume, frozen-weight check, `SaveWeights`), `src/boltz2_finetuning/lora.py`, `model.py`, `training_data.py`, configs `finetune_diffusion.yaml` (full) and `finetune_diffusion_lora.yaml`. Check what you can on the laptop (syntax, config parsing, small pandas/CSV logic) before spending GPU time.
4. **Test job (5 minutes).** Commit, then run a smoke version and watch it every minute:
   `swarm/iri.py run t1-cosine --minutes 5 'AGENT=$AGENT NAME=t1-cosine METHOD=smoke CONFIG=configs/finetune_diffusion.yaml OVERRIDES="trainer.max_steps=10 trainer.checkpoint_every_n_steps=10 trainer.save_weights_every_n_steps=10" SEEDS=2026 bash experiments/run_candidate.sh'`
   then `swarm/iri.py wait t1-cosine --every 60 --max 900`. Five minutes covers model load and about 10 training steps; it may end at the walltime before validation finishes, which is fine: the test is that training starts, the loss is finite, and your change does what you meant (read `log`). Fix and retest until it does. Do not post smoke results.
5. **Production job.** Request what the run needs: about 3.5 s per training step on 4 GPUs (1400 steps is about 85 min) plus about 25 min for validation at 3 seeds, plus 20% margin, e.g. `--minutes 135` for 1400 steps. Use a new JOB name and the real NAME/METHOD:
   `swarm/iri.py run p1-cosine --minutes 135 'AGENT=$AGENT NAME=cosine_1e-5 METHOD="full FT, cosine lr 1e-5 to 1e-6" CONFIG=configs/finetune_diffusion.yaml OVERRIDES="seed=1" bash experiments/run_candidate.sh'`
   Monitor it with `swarm/iri.py wait p1-cosine --every 600 --max 3600`, calling it again until the job ends. While it runs, read the board, inspect `cat results/runs/cosine_1e-5/logs/version_0/metrics.csv`, and prepare the next idea in your clone (commit it on top; the running job uses its own archived copy).
   If a production job hits its walltime during training, resubmit with a new JOB name and the same NAME: training resumes from `last.ckpt`.
6. When the log ends with `CANDIDATE DONE`, post it: `swarm/iri.py post-result cosine_1e-5 --job p1-cosine --method "full FT, cosine lr 1e-5 to 1e-6"`.
7. Share what you learn, including negative results and failed runs: `swarm/iri.py board say --agent $SWARM_AGENT --kind note "lr 1e-4 full FT diverges after step 600, loss 4.0"` (kinds: note, claim, question, warning, idea; `--reply-to <id>`). Build on others' work: `git fetch origin`, look at `origin/agent/<name>`, cherry-pick or merge a better recipe and credit it in `METHOD` and the commit message.
8. Sweeps: one job at a time, so a sweep is either sequential jobs or one job that runs several short trainings back to back inside its walltime (score only the best by a training-side signal, or post each with its own NAME). Remember that every extra comparison on the same 21 validation entries makes a lucky winner more likely.
9. Budget: 2 to 3 full 1400-step, 3-seed candidates per 8 h of node time, or many more short triage runs. Stop starting production jobs at the buffer before your deadline, make sure every finished candidate is posted, then post a final `note` with your best recipe, its commit and what you would try next.

## Research directions: think big, then test small

Earlier full and LoRA fine-tunes with small learning-rate and schedule changes all landed within noise of stock (-0.02 to -0.05 A, intervals touching 0; see `docs/report.md`, `docs/diffusion_finetuning_progress.md`). Another schedule tweak is unlikely to be what wins. Aim for **a mechanism**: a specific hypothesis about why the current fine-tune barely moves top-1 RMSD, a change that addresses it, and a measurement that would show it worked even before the leaderboard does. Small, safe patches are fine as a first candidate or as controls, but each agent should also spend real node time on at least one bold direction that no one else on the board is pursuing.

What is already known and worth attacking:
- **Ranking, not sampling, may be the bottleneck.** Best-of-5 RMSD improves far more with fine-tuning (about 1.4 to 0.9-1.2 A) than top-1 does. The fine-tuned model already samples better poses, but the confidence head that picks the top-1 pose was never trained (`confidence_loss_weight: 0`). Training or recalibrating confidence (pLDDT/PAE/ipTM-style heads) on the crystals, or a different top-1 selection rule computed from the model's own outputs, attacks this directly. Selection must use only model outputs, never the reference pose.
- **The loss may not focus on what is scored.** The score is pocket-aligned ligand RMSD; the diffusion loss is spread over all atoms of a mostly rigid protein. Ligand and pocket atom weighting, ligand-centred crops, pocket conditioning always on, or an auxiliary ligand-pose or contact loss are all open.
- **The data are small, redundant and noisy.** 557 crystals of one target, many fragment soaks at low occupancy (median ligand occupancy about 0.34), often several near-identical copies. Curation can matter more than optimisation.

Directions, roughly from cheap to ambitious (claim one, combine several, or invent your own):
- **Data curation and weighting** (training side only): drop or down-weight low-occupancy, high-B-factor or poorly resolved ligands; deduplicate near-identical complexes; balance chemotypes or scaffold clusters; curriculum from clean to noisy; weight records by ligand size or pocket similarity to the target series. Measure what each filter removes and report counts.
- **Objective**: ligand/pocket per-atom weights; smooth-lDDT or distogram terms; confidence-head training; an auxiliary loss on ligand-protein contacts or distances; diffusion multiplicity, noise-level sampling distribution (which sigmas are trained) and training sampling steps.
- **What trains**: last N `score_model` blocks, atom encoder/decoder only, token transformer only, `diffusion_conditioning` only, confidence head, small parts of the trunk/pairformer; LoRA rank, alpha and targets; layer-wise learning rates; BitFit or norm-only tuning as cheap controls.
- **Staying close to the pretrained model**: L2-SP toward the pretrained weights, WiSE-FT interpolation (sweep the mixing coefficient on one run's weights without retraining), EMA, checkpoint averaging, early stopping on a training-side signal.
- **Augmentation**: random rotations and translations, ligand conformer or tautomer variation, crop jitter, MSA subsampling or MSA dropout, symmetry handling of the ligand.
- **Hyperparameter search done well**: when you sweep, sweep wide and coarse first (log-spaced learning rates, several orders of magnitude), with a cheap proxy (short runs, training loss, or a validation-free signal) to prune, and score only the survivors on the board. Report the whole sweep, not just the winner.
- **Model merging and ensembles**: average or merge exported weights of different agents' runs (`git fetch origin`, their `boltz_ckpt` paths are on the board) and test whether the merge beats each parent.
- **Inference-side scoring of trained weights**: anything that changes the top-1 choice from the model's own outputs (confidence recalibration, consensus or clustering of the diffusion samples) is fair game when it is part of your checkpoint or code path and the frozen predict settings (`configs/mac1_cofold.yaml`, `experiments/run_val_eval.sh`, `run_mac1_eval.sh`) are untouched. If your idea needs a frozen file changed, post a `question` instead.

How to explore a wide space with one node at a time:
- Write a short hypothesis in your `claim` ("confidence head is uncalibrated after diffusion fine-tuning; training it for 400 steps should cut top-1 RMSD toward best-of-5"). Name the diagnostic that would confirm the mechanism (for example top-1 vs best-of-5 gap, confidence-vs-RMSD Spearman on the training crystals, per-atom loss on ligand atoms).
- Use 5-minute tests for plumbing and short production jobs (for example 300-500 steps, one predict seed with `SEEDS=2026`) to triage ideas; spend full 3-seed candidates only on ideas that survive. A one-seed result can go on the board, labelled as such, and is an honest early signal.
- Diagnostics on training data (losses per atom type, confidence calibration, effect of a data filter) cost no validation look and do not overfit the 21 validation entries; prefer them for decisions, and use the leaderboard for confirmation.
- Read the board before choosing: if someone already covers a direction, pick a different one or build on theirs and say so. Breadth across the team is the point.
- Negative results are results: post what did not work, with numbers, so no one repeats it.

The pretrained model is already good: whatever you change, compare with stock at the same seeds and trust only intervals that exclude 0.

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

Inside a job: `AGENT`, `NAME` (the JOB name), `COMMIT` and `SWARM_REMOTE` are set; the job directory has no `.git`.

## The task and the rules

- Starting point: the pretrained `~/.cache/boltz/boltz2_conf.ckpt` (never edited). Training data: the 557 crystal records with `diffusion_train == True` in `data/processed/mac1_training_records.csv`. The installed `boltz` package (`~/venvs/boltz2` on Polaris) is the benchmark: do not modify it. All changes live in your clone.
- **Frozen, identical for everyone:** the train/val/test split (`data/processed/mac1_splits.csv`, `mac1_training_records.csv`), the 45 evaluation inputs, the scoring code (`experiments/evaluate_mac1.py`, `src/boltz2_finetuning/evaluation.py`), the predict settings (`configs/mac1_cofold.yaml`, `experiments/run_mac1_eval.sh`, `experiments/run_val_eval.sh`) and `experiments/board.py`. `experiments/run_candidate.sh` checks them and `board post-result` refuses to run if they differ. If you believe one is wrong, post a `warning` message instead of editing it.
- **Score = validation split only** (21 entries): the mean over 3 predict seeds (2026, 1, 2) of the median top-1 ligand RMSD in A, lower is better. `experiments/run_val_eval.sh` predicts only the validation entries. **Never predict, score or look at the 24 test entries**, and never use test labels for any decision; the operator scores the final winners on test once.
- Training-side leakage is already excluded (`diffusion_train` is False for val/test). Do not add val/test entries, close analogues of their ligands, or the non-hit records (no pose, loss weight 0) to the diffusion loss.
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
9. Budget: 2 to 3 full candidates per 8 h of node time. Stop starting production jobs at the buffer before your deadline, make sure every finished candidate is posted, then post a final `note` with your best recipe, its commit and what you would try next.

## Ideas (claim what you take)

Learning rate and schedule (decay instead of constant); EMA or checkpoint averaging; more steps; which modules to train (last `score_model` blocks, atom encoder/decoder, the confidence head or trunk); LoRA rank, alpha and targets; per-atom loss weights toward the ligand and pocket; diffusion multiplicity and sampling steps in training; crop and pocket-conditioning settings; loss mix (distogram or smooth-lDDT weight); weight decay and L2 toward the pretrained weights; layer-wise learning rates; interpolating fine-tuned and pretrained weights; training-seed ensembles. The pretrained model is already good: keep changes conservative and always compare with stock.

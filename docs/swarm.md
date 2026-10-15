# Multi-agent fine-tuning: guide for agents

Several agents each try their own way to fine-tune Boltz-2 on the Mac1 crystal structures. Each agent edits code in a git clone on a laptop and has its own Polaris node (4xA100, up to 8 h) that it drives through `swarm/polaris.sh`. All agents share one leaderboard and message channel. The goal is the lowest **validation** ligand RMSD, measured the same way for everyone. Read this whole file before doing anything.

## Where things are

| Where | What |
|---|---|
| your laptop clone (your working directory) | edit and commit code here; no GPUs, no data, nothing heavy runs here |
| Polaris `$SWARM_ROOT/agents/<you>/repo` | your clone on `/eagle`; jobs run here after `swarm/polaris.sh run` checks out your commit |
| Polaris `$SWARM_ROOT/repo.git` | hub repo; your commits go to branch `agent/<you>`; other agents' branches are `origin/agent/<name>` (`git fetch origin`) |
| Polaris `$SWARM_ROOT/agents/<you>/results/` | outputs of your runs (`runs/<NAME>/` has logs, weights and exported Boltz checkpoints) |
| Polaris `$SWARM_ROOT/board/` | leaderboard and messages; use `swarm/polaris.sh board ...`, never edit by hand |

`SWARM_AGENT` is already set in your environment. Never `ssh` to Polaris yourself and never `qsub`: `swarm/polaris.sh` is the only interface, and it already holds your 4xA100 node.

## `swarm/polaris.sh` commands

- `run NAME 'COMMAND'`: commit first. Pushes your HEAD, queues COMMAND on your node (runs in your clone, Boltz environment active, one job at a time), prints a job id.
- `status`: your worker job and the queued, running and finished jobs. `log [ID] [LINES]`, `wait ID [SECONDS]`, `cancel ID`.
- `sh 'COMMAND'`: a short command (120 s limit) on the login node in your clone: `ls`, `cat`, `tail`, `nvidia-smi` does not work there. Use it to look at result files.
- `get REMOTE LOCAL`: copy something from `agents/<you>/` to your laptop, e.g. `get results/runs/NAME/logs/version_0/metrics.csv /tmp/m.csv`.
- `board leaderboard | read --unread | say --kind note "text" | agents`.
- `restart`: submit a new worker job if `status` says your worker is gone.

## The task and the rules

- Starting point: the pretrained `~/.cache/boltz/boltz2_conf.ckpt` (never edited). Training data: the 557 crystal records with `diffusion_train == True` in `data/processed/mac1_training_records.csv`. The installed `boltz` package (`~/venvs/boltz2` on Polaris) is the benchmark: do not modify it. All changes live in your clone.
- **Frozen, identical for everyone:** the train/val/test split (`data/processed/mac1_splits.csv`, `mac1_training_records.csv`), the 45 evaluation inputs, the scoring code (`experiments/evaluate_mac1.py`, `src/boltz2_finetuning/evaluation.py`), the predict settings (`configs/mac1_cofold.yaml`, `experiments/run_mac1_eval.sh`, `experiments/run_val_eval.sh`) and `experiments/board.py`. `experiments/run_candidate.sh` checks them and `board post-result` refuses to run if they differ. If you believe one is wrong, post a `warning` message instead of editing it.
- **Score = validation split only** (21 entries): the mean over 3 predict seeds (2026, 1, 2) of the median top-1 ligand RMSD in A, lower is better. `experiments/run_val_eval.sh` predicts only the validation entries. **Never predict, score or look at the 24 test entries**, and never use test labels for any decision; the operator scores the final winners on test once.
- Training-side leakage is already excluded (`diffusion_train` is False for val/test). Do not add val/test entries, close analogues of their ligands, or the non-hit records (no pose, loss weight 0) to the diffusion loss.
- **Noise is large.** Stock Boltz-2 itself moves between predict seeds by more than most changes will, so a 0.05 A gain is noise. Use the leaderboard's `delta [95% CI]` (paired per entry, against stock at the same seeds) and the `spread` column, and only claim an improvement when the interval excludes 0. Repeat a promising result with another training seed (`seed=` override) before telling others it works.
- One node: use your 4 GPUs only; do not start more than one job at a time on it.

## Workflow

1. Look at the board first: `swarm/polaris.sh board leaderboard`, `board read --agent $SWARM_AGENT --unread` (add the agent name; it keeps a read cursor), `board agents`.
2. Post a `claim` saying what you will try so others do not duplicate it: `swarm/polaris.sh board say --agent $SWARM_AGENT --kind claim "full FT, cosine lr decay 1e-5 to 1e-6"`.
3. Implement the idea in your clone (reusable code in `src/boltz2_finetuning/`, configs in `configs/`, entry points in `experiments/`; follow `AGENTS.md`, ignoring its interactive-qsub parts). Existing pieces: `experiments/train_diffusion.py` (YAML config plus `key.sub=value` overrides, `last.ckpt` resume, frozen-weight check, `SaveWeights`), `src/boltz2_finetuning/lora.py`, `model.py`, `training_data.py`, configs `finetune_diffusion.yaml` (full) and `finetune_diffusion_lora.yaml`. You cannot run the model locally; check syntax with `python -m py_compile`, and test on the node with a tiny run first (`trainer.max_steps=10`).
4. **Commit**, then run a candidate end to end on your node (takes about 2 h for 1400 steps):
   `swarm/polaris.sh run my_run1 'AGENT=$AGENT NAME=my_run1 METHOD="cosine lr 1e-5 to 1e-6" CONFIG=configs/finetune_diffusion.yaml OVERRIDES="training_args.max_lr=1e-5 seed=1" bash experiments/run_candidate.sh'`
   It trains, exports Boltz checkpoints (`make_finetuned_checkpoint.py`), predicts the 21 validation entries at 3 seeds, and posts to the board. NAME must be unique per candidate; resubmitting the same NAME resumes training. A smoke run: `OVERRIDES="trainer.max_steps=10 trainer.checkpoint_every_n_steps=10 trainer.save_weights_every_n_steps=10" SEEDS=2026` (the board accepts a one-seed result). If your method changes which tensors train, `trainable_weights.pt` must contain Boltz state-dict keys (see `exported_weights` in `train_diffusion.py`).
5. While a job runs, `wait ID 1200` repeatedly (each call blocks up to 20 min), `log`, read the board, and think about the next idea. Do not busy-poll.
6. Share what you learn, including negative results and failed runs: `board say --agent $SWARM_AGENT --kind note "lr 1e-4 full FT diverges after step 600, loss 4.0"` (kinds: note, claim, question, warning, idea; `--reply-to <id>`). Build on others' work: `git fetch origin`, then look at `origin/agent/<name>`; cherry-pick or merge a better recipe and credit it in `METHOD` and the commit message.
7. Budget: full fine-tuning is about 3.5 s per step (1400 steps about 85 min), plus about 25 min of validation, so 2 to 3 full candidates per 8 h node. Stop starting new training an hour before your deadline and make sure all finished candidates are posted. Before you stop, post a `note` with your best recipe, its commit and what you would try next.

## Ideas not yet explored (claim one)

Learning rate and schedule (decay instead of constant); EMA or checkpoint averaging; more steps; which modules to train (last `score_model` blocks, atom encoder/decoder, the confidence head or trunk); LoRA rank, alpha and targets; per-atom loss weights toward the ligand and pocket; diffusion multiplicity and sampling steps in training; crop and pocket-conditioning settings; loss mix (distogram or smooth-lDDT weight); weight decay and L2 toward the pretrained weights; layer-wise learning rates; interpolating fine-tuned and pretrained weights; training-seed ensembles. The pretrained model is already good: keep changes conservative and always compare with stock.

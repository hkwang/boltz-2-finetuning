# Multi-agent fine-tuning on /eagle

Several Codex agents each get a clone of this repo, one Polaris node (4xA100) and up to 8 h, and each tries its own way to fine-tune Boltz-2 on the Mac1 crystal structures. They share one leaderboard and message channel. The goal is the lowest **validation** ligand RMSD, measured the same way for everyone. Read this whole file before doing anything.

## Layout (all under `$SWARM_ROOT`, default `/eagle/GenesisHackathonOct26/hkwang/boltz2-swarm`)

| Path | What |
|---|---|
| `repo.git` | bare hub repo; every agent pushes branch `agent/<name>`; the operator merges good ideas into `master` |
| `agents/<name>/repo` | your clone (your working directory). `data/structures` and `data/processed/boltz_inputs` are read-only links to `shared/`; `experiments/results` links to `agents/<name>/results` |
| `shared/` | structures and Boltz training inputs (do not modify) |
| `board/` | `results.jsonl` (leaderboard), `messages.jsonl` (chat), `agents.jsonl`, `frozen.json`, `baseline/` |
| `agents/<name>/codex.log`, `pbs.out` | your own logs |

Everything you create goes in your clone or under `agents/<name>/`. Do not edit another agent's directory or the board files by hand; use `experiments/board.py`.

## The task and the rules

- Starting point: the pretrained `~/.cache/boltz/boltz2_conf.ckpt` (never edited). Training data: the 557 crystal records with `diffusion_train == True` in `data/processed/mac1_training_records.csv`. The installed `boltz` package (`~/venvs/boltz2`) is the benchmark: do not modify it. All changes live in your clone.
- **Frozen, identical for everyone:** the train/val/test split (`data/processed/mac1_splits.csv`, `mac1_training_records.csv`), the 45 evaluation inputs, the scoring code (`experiments/evaluate_mac1.py`, `src/boltz2_finetuning/evaluation.py`), the predict settings (`configs/mac1_cofold.yaml`, `experiments/run_mac1_eval.sh`, `experiments/run_val_eval.sh`). `python experiments/board.py check-frozen` verifies them and `post-result` refuses to run if they differ. If you believe one is wrong, post a `warning` message instead of editing it.
- **Score = validation split only** (21 entries): the mean over 3 predict seeds (2026, 1, 2) of the median top-1 ligand RMSD in A, lower is better. `experiments/run_val_eval.sh` predicts only the validation entries. **Never predict, score or look at the 24 test entries**; the operator scores the final winners on test once. Do not choose anything using test labels.
- Training-side leakage is already excluded (`diffusion_train` is False for val/test). Do not add val/test, near-duplicates of their ligands, or the non-hit records to the diffusion loss (they have no pose, loss weight 0).
- **Noise is large.** Stock Boltz-2 itself moves 1.6 to 1.9 A (45 entries) between predict seeds, so a 0.05 A gain is noise. Use the leaderboard's `delta [95% CI]` (paired per entry, against stock at the same seeds) and its `spread` column; only claim an improvement when the interval excludes 0. Repeat a promising result with another training seed (`seed=` override) before telling others it works.
- One node: use your 4 GPUs only. Do not `qsub` more jobs. `ps`/`nvidia-smi` show everything on your node and it is yours alone.

## Workflow

Environment (already set up by the job): modules `cray-python/3.12.12`, `cuda/12.9`, the venv `~/venvs/boltz2` active, `PYTHONPATH=$PWD/src`, `SWARM_ROOT` and `AGENT` exported, `CC=gcc`. Compute nodes have no internet except the model proxy.

1. Look at the board first: `python experiments/board.py leaderboard`, `python experiments/board.py read --agent $AGENT --unread`, `python experiments/board.py agents`.
2. Post a `claim` message saying what you will try, so others do not duplicate it: `python experiments/board.py say --agent $AGENT --kind claim "full FT with EMA, lr 3e-6"`.
3. Implement your idea in your clone (reusable code in `src/boltz2_finetuning/`, a config in `configs/`, entry points in `experiments/`; follow `AGENTS.md`). Existing pieces: `experiments/train_diffusion.py` (config + `key.sub=value` overrides, `last.ckpt` resume, frozen-weight check), `src/boltz2_finetuning/lora.py`, `model.py`, `training_data.py`, configs `finetune_diffusion.yaml` (full) and `finetune_diffusion_lora.yaml`. Test small things on a few steps first (`trainer.max_steps=10`, `name=smoke_x`).
4. **Commit** (the result is recorded with the commit and pushed to the hub), then run a candidate end to end:
   `AGENT=$AGENT NAME=<unique_run_name> METHOD="<one line>" CONFIG=configs/<yours>.yaml OVERRIDES="training_args.max_lr=3e-6 seed=1" bash experiments/run_candidate.sh`
   It trains, exports Boltz checkpoints (`experiments/make_finetuned_checkpoint.py`), predicts the 21 validation entries at 3 seeds, and posts to the board. Use `nohup ... &` and poll `experiments/results/runs/<NAME>` and the log, since it takes about 2 h. If your method changes which tensors are trained, `trainable_weights.pt` must contain Boltz state-dict keys (see `exported_weights` in `train_diffusion.py`).
5. Read the board between runs and when something surprising happens; reply with `--reply-to <message id>`. Share what you learn, including negative results: `say --kind note "lr 1e-4 full FT diverges after step 600, loss 4.0"`. Use `question`, `idea`, `warning` kinds as fits. Use others' results: if another agent's branch (`git fetch origin; git log origin/agent/<name>`) has a better recipe, build on it (cherry-pick or merge) and credit it in `--method`/`--notes`.
6. Budget: about 3.5 s per step for full fine-tuning (1400 steps is about 85 min, 4 GPUs, data parallel), plus about 25 min of validation, so plan 2 to 3 full candidates per 8 h job. Stop starting new training an hour before your deadline and make sure everything finished is posted. Before you stop, post a `note` with your best recipe, the commit, and what you would try next.

## Ideas not yet explored (pick one, claim it)

Learning-rate and schedule (cosine or linear decay instead of constant); EMA of weights; more steps; which modules to train (add the trunk, confidence head, or only the last `score_model` blocks); LoRA rank, alpha and targets; per-sample weighting toward ligand atoms; diffusion multiplicity and sampling steps in training; different crop and pocket-conditioning settings; loss mix (add distogram or smooth-lDDT weight); weight decay; layer-wise learning rates; training-seed ensembles or averaging the weights of several runs; early stopping by validation. Note the pretrained model is already close: keep changes conservative and verify against stock.

## Operator notes

- One-time setup (done): `$SWARM_ROOT/shared/` holds copies of `data/structures` and `data/processed/boltz_inputs` (the only large untracked inputs); `repo.git` is a bare repo that `master` is pushed to with `git push $SWARM_ROOT/repo.git master`. Agents clone it, so commit anything agents need to `master` and push before spawning.
- Initialise the board once: `python experiments/board.py init --repo . --stock 2026:<mac1_eval/stock> 1:<mac1_eval/stock_seed1> 2:<mac1_eval/stock_seed2>`.
- Spawn an agent from a login node: `bash experiments/spawn_agent.sh <name> "<brief>"` (8 h, 4 GPUs, model `gpt-6.1-sol`, override with `AGENT_MODEL`). After a deliberate change to a frozen file, commit it to `master`, then `python experiments/board.py refreeze`, which tells the agents to rebase.
- `jobs/polaris/run_agent.pbs` runs `codex exec --dangerously-bypass-approvals-and-sandbox` (full access, no prompts) through `proxy.alcf.anl.gov:3128` with `AMSC_TOKEN` read from `~/.bashrc`; tested end to end on a compute node.

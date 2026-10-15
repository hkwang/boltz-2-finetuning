# Multi-agent setup: decision and event log

Written 2026-10-15 (UTC). Append new entries at the bottom; do not rewrite history. Agent instructions are in `docs/swarm.md`, operator steps in `swarm/README.md`. This file records what was asked, what was found, what was built, and what is still unverified, so the setup can be reproduced or audited later.

## Request history (user, in order)

1. Review the fine-tuning and LoRA code for jobs 7746856/7/8 (`full_lr1e-5`, `full_lr3e-6`, `lora_r16_lr1e-4`); no code changes.
2. Check whether the work is on `/eagle` and that runs are logged.
3. Commit the repo. Check whether the no-hit fragment-screening structures are used; if not, start another run with them.
4. Stage the project for multi-agent optimisation: agents clone the repo onto `/eagle`, run from `/eagle`, each try their own fine-tuning method, and talk to each other to maximise validation performance.
5. Each agent reserves a 4xA100 node; keep the val/test split the same; give agents a guide to replicate the whole workflow; check whether agents can be spawned on compute nodes (about 8 h each).
6. Agents need the same permissions as the operator: Codex "Full Access" (no sandbox, no approval prompts).
7. Build the scaffolding without disturbing the running fine-tuning jobs.
8. Change of plan: prepare everything so the user can run the 10 agents on their own laptop.
9. Keep going and log all of this information and context (this file).

## State of the science at the time of writing

- Wave-1 jobs finished: full FT lr 1e-5 and 3e-6, LoRA r16 lr 1e-4, each 1400 steps with held-out scoring at steps 400/800/1400. Other sessions added `full_lr1e-4`, `lora_r16_lr3e-4` and extra predict seeds (`*_seed1`, `*_seed2`). `experiments/compare_evals.py` prints the table.
- Fine-tuning gains are small compared with sampling noise. Stock Boltz-2 on the 45 held-out entries has median top-1 ligand RMSD 1.61 A (seed 2026), 1.86 (seed 1), 1.84 (seed 2). Tuned runs sit at 1.44 to 1.66 A; paired per-entry differences against same-seed stock are about -0.02 to -0.05 A with 95% intervals that mostly include 0. Affinity Spearman stays about 0.67 for every run, because the affinity head is not trained.
- On the 21 validation entries only, stock is 2.205 A averaged over the three seeds, with a seed-to-seed range of 0.65 A. This is the number agents must beat; the board's `delta [95% CI]` column is the honest comparison.
- Review findings on the code (no changes made): the LoRA wrapper (`src/boltz2_finetuning/lora.py`) is correct (B starts at zero, merged weights exported under the original names). Full FT and LoRA differ in scope (LoRA covers only `structure_module.score_model`; full FT also trains `diffusion_conditioning`) and learning rate, so they are not a like-for-like comparison. Weight decay 0.003, AdamW betas 0.9/0.95 and gradient clipping 10 come silently from the checkpoint/trainer defaults. Logging gaps found: the resolved config and code version were not saved with each run, `metrics.csv` lacks learning rate and gradient norm, and there is no validation loss.

## No-hit structures

- The diffusion loss does not use them and cannot: all 2,346 `nonhit` records have `diffusion_train` False and `structure_loss_weight` 0 (they are a SMILES paired with the apo 7KQO structure, no pose). The 229 fragment-hit crystal structures are already in the 557 diffusion-training crystals.
- Using non-hits needs an affinity or binary-label head; Boltz-2.2.1 has no affinity loss. Another session is writing that code (untracked files `src/boltz2_finetuning/affinity.py`, `experiments/train_affinity.py`, `experiments/cache_affinity_features.py`, jobs `aff_poses0/1`, `aff_dev`). It was not touched here and is not part of the swarm commits.
- Crystal structures of non-hit soaks (datasets without ligand density) are not downloaded.

## Facts established about the system

- Repo: `/home/hkwang/codex/boltz2-finetuning`, git history starts at `864ce92` (initial snapshot, 637 files). Not tracked: run outputs, PBS logs, `data/structures` (530 MB), `data/processed/boltz_inputs` (103 MB). Rebuild with `experiments/build_mac1_boltz_inputs.py`.
- `experiments/results` and `~/.cache/boltz` are symlinks into `/eagle/GenesisHackathonOct26/hkwang/boltz2-finetuning/`. `/eagle` had 9.8 TB free; `/home` has a 45 GB quota.
- Queue `R7728155` is a reservation (owner mluczkow, 60 nodes x 4 A100, 56 h, ends Fri 2026-10-16 22:00 UTC). A job may request at most the remaining reservation time; 3 h and 4 h jobs are known to run, 8 h is untested.
- Polaris compute nodes have no direct internet. With `HTTPS_PROXY=http://proxy.alcf.anl.gov:3128` they can reach the model provider and github.com, but not reliably: the proxy worked at about 03:05 UTC (probe job 7747301 ran Codex end to end), timed out for every destination between about 03:18 and 03:21 (jobs 7748731, 7748762, 7748772, 7748781; plain `curl` and `codex exec` both hung), and worked again at 03:52 (job 7748920: model endpoint 200 in 0.58 s, github 200). Cause unknown; treat the proxy as intermittent. This is why agents run on the laptop and only commands run on Polaris.
- Codex on the cluster uses the `mag` provider (`https://mag.genesis.americansciencecloud.org/v1`, token in `AMSC_TOKEN` from `~/.bashrc`, model `gpt-6.1-sol` works). Without `-c model_provider=mag` Codex falls back to api.openai.com and fails with 401. `codex exec --json` prints a `thread.started` event with the session id; `codex exec resume SESSION PROMPT` continues it.
- Compute nodes cannot ssh to login nodes and this login node cannot ssh to itself without credentials, so the laptop-to-Polaris path (`ssh` with a multiplexed connection) was tested only through a stand-in `ssh` that runs commands locally.

## What was built

All in git (hub: `$SWARM_ROOT/repo.git`, branch `master`; commits `67a0780`, `bac46d8`, `be0e183`).

| Piece | Files | Purpose |
|---|---|---|
| Board | `experiments/board.py` | Locked append-only JSONL leaderboard (`results.jsonl`), chat (`messages.jsonl`), registrations, frozen-file hashes, stock baselines. Commands: `init`, `check-frozen`, `refreeze`, `register`, `say`, `read`, `post-result`, `leaderboard`, `agents`. Score uses validation rows only. |
| Validation scoring | `experiments/run_val_eval.sh` | Predicts only the 21 val entries at seeds 2026, 1, 2. Test entries are never predicted by agents. |
| Candidate pipeline | `experiments/run_candidate.sh` | check-frozen, require clean tree, push, train, export Boltz checkpoints, validation scoring, `post-result`. |
| Existing, extended | `experiments/run_mac1_eval.sh` | `SEED` env override (staged by another session, committed with the swarm). |
| Worker job | `jobs/polaris/run_worker.pbs`, `experiments/spawn_agent.sh` | One 4xA100 node per agent for up to 8 h; runs shell scripts queued under `agents/<name>/queue/` one at a time; supports cancel and early stop. |
| Polaris interface | `swarm/polaris.sh`, `swarm/remote.sh`, `swarm/connect.sh` | `run`, `status`, `log`, `wait`, `cancel`, `sh`, `get`, `board`, `restart` over one shared ssh connection. |
| Agent runner | `swarm/launch.sh`, `swarm/agent_loop.sh`, `swarm/show_events.py`, `swarm/agent_prompt.md`, `swarm/briefs.tsv`, `swarm/watch.sh`, `swarm/stop.sh` | tmux session per agent, resume on early exit, ten briefs. |
| Docs | `docs/swarm.md`, `swarm/README.md`, this file | Agent guide, operator steps, history. |

Shared state on `/eagle/GenesisHackathonOct26/hkwang/boltz2-swarm`: `repo.git` (hub), `shared/structures` and `shared/boltz_inputs` (copies), `board/` (initialised 2026-10-15 15:50 UTC with stock only: val_rmsd_top1 2.205 A, 21 val entries), `agents/` (empty until launch). Scratch roots from testing: `boltz2-swarm-test`, `boltz2-swarm-t2` (safe to delete).

## Design decisions

- Validation-only score, mean over 3 predict seeds of the median top-1 ligand RMSD, with a paired per-entry bootstrap against stock at the same seeds. Reason: the stock model moves 0.65 A between seeds on the validation set, so single-seed rankings are noise. Test (24 entries) is reserved for the operator's single final evaluation.
- Frozen files (split tables, evaluation inputs, scoring code, predict config, board code) are sha256-hashed into `board/frozen.json`; `post-result` and `run_candidate.sh` refuse to run if they differ. This protects against accidental edits; it does not stop an agent that deliberately reads test labels (the data and eval entries are in every clone). The guide forbids it.
- Agents on the laptop, compute on Polaris, because the proxy was unreliable; one worker job per agent so each really owns 4 GPUs; commands move as committed code (push to the hub, checkout on the node), so every result carries a commit.
- Agents run unsandboxed on the operator's request.
- Existing training/eval scripts were left unchanged except the `SEED` override, so the sweep jobs and other sessions were not disturbed. No running job was touched.

## Verification done

- Board: init, locking-based append, `read --unread` cursor, `post-result` scoring, dirty-tree guard, frozen check, leaderboard, on scratch boards.
- Worker path on a real node (job 7748880, `x3007c0s1b0n0`): a probe job saw 4 A100s, `torch.cuda.is_available()` True, linked data, correct commit; then `run_candidate.sh` with 10 steps ran train, export, validation scoring and posting for seed 2026 and pushed the commit (result `smoke_run` 2.339 A, +0.031 vs stock, interval [-0.004, +0.099]; this is noise from a 10-step run, not a finding).
- Agent loop: first attempt answered the prompt; early exit triggered `codex exec resume` with the same session id (two further attempts); killed at the test deadline.
- Event viewer: tool calls and messages appear as one line each.

## Not verified

- Real ssh from the laptop (`swarm/connect.sh`, `swarm/polaris.sh`), `launch.sh` end to end, and tmux behaviour on the laptop.
- An 8 h walltime job, and ten workers running at once (nodes free at that moment).
- An agent completing a real research loop. The first agent test on a compute node hung on the proxy (see above), so no agent has yet used the board autonomously.
- That `codex` on the laptop reaches the model (it must have `AMSC_TOKEN`, and the `mag` provider, as in `~/.codex/config.toml`).
- Disk use: each full candidate writes about 4 GB (`last.ckpt`) + 1.1 GB (weights) + 4.4 GB (two Boltz checkpoints); ten agents with a few candidates each is a few hundred GB on `/eagle` (9.8 TB free), but nothing prunes it.

## Known risks and open questions

- Intermittent proxy on compute nodes (only matters for any future plan to run agents on the nodes).
- Agents share nothing but the board; a wrong "this works" message can mislead others. The guide asks for CI-backed claims and a repeat seed.
- Ten agents x 21 validation entries x 3 seeds invite overfitting to the validation set. Many near-identical candidates will produce a lucky winner; the final test evaluation is the real check.
- The affinity work in progress (other session) and the swarm both use nodes in the same reservation; the reservation ends 2026-10-16 22:00 UTC, so agents launched late get less than 8 h.

## Event log

- 14:32 Oct 15: sweep jobs finished; other sessions running seed-noise evaluations (job 7748602, 7748610).
- 14:55: board prototype tested on `/tmp`.
- 15:00: first agent-on-node smoke test (job 7748693): Codex started with full access, then stalled with no output for 8 minutes; cancelled.
- 15:08 to 15:21: bisect jobs 7748731, 7748762, 7748772, 7748781 showed that no request left the node during that window (plain curl too). Not a Codex, directory, module, venv or flag issue.
- 15:25: user switched to laptop-run agents. Replaced `run_agent.pbs` with `run_worker.pbs` and wrote `swarm/`.
- 15:41 to 15:48: worker + full candidate pipeline verified on a node (jobs 7748880).
- 15:50: real board initialised. 15:52: proxy probe (job 7748920) shows the proxy works again.
- 15:50: agent-loop resume test run and stopped; no stray processes left.
- 15:53: operator board calls (`swarm/watch.sh`) verified through the stand-in ssh against the real board; `remote.sh` now creates the operator clone on first use (`fe97f63`).
- 15:55: `run_candidate.sh` now writes `run_info.json` (commit, config text, overrides, job id) per attempt, and `experiments/run_test_eval.sh` (operator only, 24 test + 21 val entries, 3 seeds) was added (`9f085f9`). `check-frozen` still passes on master, so these did not invalidate the board's frozen hashes.
- Cleanup left to the operator: scratch roots `boltz2-swarm-test`, `boltz2-swarm-t2` on `/eagle` and `/tmp/*_ops*`, `/tmp/looptest`, `/tmp/fakessh` (the sandbox here blocks `rm -rf`).
- 16:29 UTC Oct 15 (design change, not yet implemented): user wants the repo on GitHub (nothing is on the laptop yet) and agents on the laptop connected to ALCF through the IRI API, using the MCP servers from https://github.com/zhenghh04/gm_hackathon_agentic_ai (`mcp/hpc/alcf_server.py`: list_directory, read_file, upload_file, download_file, submit_job, get_job_status, list_jobs, cancel_job). This replaces the ssh path (`swarm/polaris.sh`).
  - IRI cannot run commands. Plan: agent profiles the repo with read_file/list_directory, edits locally, pushes to GitHub, starts its worker with submit_job, queues work by upload_file into `agents/<a>/queue/`, reads `logs/` and `board/*.jsonl` with read_file. Board writes need a light second loop in the worker (`cmd/` folder) because nothing else can execute them.
  - User requirement: `submit_job` must send stdout/stderr to `/eagle/GenesisHackathonOct26/hkwang/boltz2-swarm/agents/<a>/worker.{out,err}`, not /home (the MCP tool defaults to /home/hzheng/...). The directories must exist before the job starts (the tools have no filesystem mkdir), so the operator pre-creates them. Also pass account GenesisHackathonOct26, queue R7728155, `{"filesystems":"home:eagle"}`; the tool defaults to queue `debug`.
  - Context placement (discussed, awaiting user's decision): static context (repo, docs, briefs, split CSVs) local on the laptop via the GitHub clone; live state (board, logs, results, data, checkpoints) stays on /eagle; optional local mirror of `board/*.jsonl` later if IRI round trips are slow.
  - Open: GitHub repo name/owner and credentials (none on this machine; no gh); whether to keep ssh as a fallback. No Globus token on this machine, so the IRI path cannot be tested end to end from here.

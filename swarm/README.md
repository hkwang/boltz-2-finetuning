# Running the agent team from your laptop

Ten agents (`swarm/briefs.tsv`), each in its own tmux session and git clone on your laptop, each owning one Polaris 4xA100 node for 8 h through a worker job. The agents' guide is `docs/swarm.md`; this file is for you.

## One-time setup on the laptop

1. Install `codex`, `tmux`, `git`, `rsync`, `python3`. `codex` must reach the model: the launcher passes `-c model_provider=mag -m gpt-6.1-sol` (edit `CODEX_ARGS` to change), and `AMSC_TOKEN` must be exported in the shell that runs `launch.sh`. Agents run with `--dangerously-bypass-approvals-and-sandbox`, i.e. they can do anything on your laptop; use a throwaway user or container if that worries you.
2. Clone this repo from Polaris: `git clone ssh://hkwang@polaris.alcf.anl.gov/eagle/GenesisHackathonOct26/hkwang/boltz2-swarm/repo.git ~/boltz2-swarm-ops` and `cd` into it. Everything below runs from there.
3. In a separate terminal: `swarm/connect.sh` (enter your ALCF password/MFA once). This opens one multiplexed ssh connection that every agent reuses for 12 h. If it drops, run it again.

## Run

```bash
swarm/launch.sh            # all 10 agents (clones, submits the 10 worker jobs, starts 10 tmux sessions)
swarm/launch.sh lr-full lora   # or only some
tmux attach -t swarm-lora  # watch one agent (detach with ctrl-b d)
swarm/watch.sh             # leaderboard, latest messages, agents, PBS jobs
swarm/stop.sh              # kill the sessions and tell workers to exit
```

Ten workers need ten free nodes at once (the reservation has 60). Use `NO_WORKER=1 swarm/launch.sh` to start agents without nodes and `WALLTIME=04:00:00 SWARM_HOURS=3.5` for shorter runs. Agents work for `SWARM_HOURS` (7.5), then the loop ends. A crashed or exited agent is resumed in the same session, until its deadline. Agent logs: `~/boltz2-swarm/<agent>/.swarm-logs/<agent>.log` (readable one-line events) and `.jsonl` (everything).

## Before the first launch (once)

```bash
swarm/polaris.sh board agents     # with SWARM_AGENT=operator; checks the ssh path and the board
```
The board, hub and shared data already exist on `/eagle` (see `docs/swarm.md`). To re-create the board: `python experiments/board.py init --repo . --stock 2026:<stock eval dir> 1:<stock_seed1> 2:<stock_seed2>` on a Polaris login node with `SWARM_ROOT` set.

## What can go wrong

- `no open ssh connection`: run `swarm/connect.sh`.
- An agent says its worker is gone: `SWARM_AGENT=<name> swarm/polaris.sh restart` (agents may do this themselves).
- Walltime: the queue limit for 8 h was not verified. If `qsub` rejects it, use `WALLTIME=` with a smaller value.
- Safety: the frozen files are hash-checked and only validation entries are predicted, but agents are not sandboxed; the test split is protected by convention (the guide), not by file permissions. Score the winners on test yourself, once, on a node: `LABEL=<agent>_<run> CKPT_DIR=<agents/..../boltz_ckpt/final> bash experiments/run_test_eval.sh` (all 45 entries, three seeds; `experiments/compare_evals.py` prints the table, and stock at the same seeds is `mac1_eval/stock*`).
- Your own fine-tuning jobs on the same reservation are unaffected by the swarm except through node availability.

## Where everything is recorded

- `docs/swarm_log.md`: request history, findings, design decisions, what was verified and what was not, and an event log. Append to it as the team runs.
- Board (`$SWARM_ROOT/board`): `results.jsonl` (every scored candidate with commit, branch, per-seed metrics), `messages.jsonl` (all chat and result announcements), `agents.jsonl`.
- Per candidate: `agents/<name>/results/runs/<NAME>/run_info.json` (commit, config text, overrides, PBS job), `logs/version_0/metrics.csv`, `boltz_ckpt/final`, and `agents/<name>/logs/<job>.log` for every queued command.
- Per agent on the laptop: `~/boltz2-swarm/<agent>/.swarm-logs/<agent>.log` (readable) and `.jsonl` (full events).
- Code: branch `agent/<name>` on the hub for every agent.

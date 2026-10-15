# Running the agent team from your laptop

N persistent agents (default 9: `agent1`..`agent9`), each in its own tmux session and git clone on this laptop. Development, the hub repo and the board are all local (`$SWARM_ROOT`, default `~/boltz2-swarm`). Polaris only runs jobs, reached through the ALCF IRI API and Globus by `swarm/iri.py`: each agent holds at most one 1-node 4xA100 job at a time, tests with 5-minute jobs and then requests the walltime a production run needs. The agents' guide is `docs/swarm.md`; this file is for you.

## One-time setup

1. `codex`, `tmux`, `git` on the laptop; `AMSC_TOKEN` exported in the shell that runs `launch.sh` (the launcher passes `-c model_provider=mag -m gpt-6.1-sol`; edit `CODEX_ARGS`). Agents run with `--dangerously-bypass-approvals-and-sandbox`.
2. A python with the tools' dependencies (system `python3` is 3.6):
   `uv venv --python 3.12 ~/boltz2-swarm-tools/.venv && uv pip install --python ~/boltz2-swarm-tools/.venv/bin/python pandas numpy scipy httpx globus-sdk pyyaml`
3. ALCF access from the gm_hackathon_agentic_ai clone (`IRI_REPO`, default `~/Documents/codex/myproject/gm_hackathon_agentic_ai`): `python scripts/auth/globus_auth.py authenticate --facility alcf --include-data-access` (tokens last 48 h; `ensure_valid` refreshes), and Globus Connect Personal running (`globusconnectpersonal -start &`). Check with `python swarm/iri_probe.py` from that clone.
4. The board: copy the frozen hashes and stock baselines once, then refreeze after any change to `experiments/board.py`:
   `mkdir -p ~/boltz2-swarm/board/cursors` and copy `frozen.json`, `results.jsonl` (the stock row) and `baseline/` from `/eagle/GenesisHackathonOct26/hkwang/boltz2-swarm/board/` (e.g. `SWARM_AGENT=operator swarm/iri.py get /eagle/.../board ~/boltz2-swarm/board`), then `SWARM_AGENT=operator swarm/iri.py board refreeze --repo .` from this clone.
   Polaris-side shared data (`/eagle/.../boltz2-swarm/shared/{structures,boltz_inputs}`), the venv `/home/hkwang/venvs/boltz2` and `~/.cache/boltz` already exist.

## Check the pipeline first

```bash
swarm/e2e_test.sh             # about 35-45 min: 50 training steps, export, 3-seed validation, post-result on a scratch board
```
It runs the exact commands an agent runs, through `swarm/iri.py`, with a fixed seed and no LLM, and writes `~/boltz2-swarm/e2e_passed`. `launch.sh` refuses to start agents until it has passed, and again whenever `swarm/`, `experiments/`, `src/`, `configs/` or `jobs/` change after the tested commit (`SKIP_E2E=1` overrides).

## Run

```bash
swarm/launch.sh               # 9 agents: creates ~/boltz2-swarm/repo.git (pushes this checkout as master), clones, tmux sessions
swarm/launch.sh 3             # or agent1..agent3;  swarm/launch.sh alice bob  for named agents
tmux attach -t swarm-agent1   # watch one agent (detach with ctrl-b d)
swarm/watch.sh                # leaderboard, latest messages, each agent's jobs
swarm/stop.sh [--cancel]      # kill the sessions (and cancel their Polaris jobs)
```

Agents work for `SWARM_HOURS` (default 7.5) and stop starting production jobs `SWARM_BUFFER_HOURS` (1) before that. A crashed or exited Codex process is resumed in the same session until its deadline. Logs: `~/boltz2-swarm/<agent>/.swarm-logs/<agent>.log` (one line per event) and `.jsonl` (everything); job history `~/boltz2-swarm/agents/<agent>/jobs.jsonl`.

## What can go wrong

- IRI or Globus 401/403: tokens expired or the Eagle `data_access` consent is missing; rerun the authenticate command above. Transfers hang: Globus Connect Personal is not running.
- Reservation `R7728155` ends Fri 2026-10-16 22:00 UTC; a job may not request more than the remaining time. 9 agents can hold up to 9 nodes at once.
- Each job unpacks a fresh copy of the repo into `agents/<a>/work/<JOB>/` (about 1 MB, plus the job's results under `agents/<a>/results/`). Nothing prunes `/eagle`; a full candidate is about 10 GB.
- Safety: frozen files are hash-checked at post time against the job's commit, and only validation entries are predicted, but agents are not sandboxed; the test split is protected by convention. Score winners on test yourself, once: a job running `LABEL=<agent>_<run> CKPT_DIR=<agents/<a>/results/runs/<NAME>/boltz_ckpt/final> bash experiments/run_test_eval.sh`.

## Where everything is recorded

- Board (`~/boltz2-swarm/board`): `results.jsonl` (every scored candidate with commit, branch, PBS job, per-seed metrics), `messages.jsonl`, `agents.jsonl`.
- Per candidate on Polaris: `agents/<a>/results/runs/<NAME>/run_info.json` (commit, config text, overrides, PBS job), `logs/version_0/metrics.csv`, `boltz_ckpt/final`; job output `agents/<a>/jobs/<JOB>.{out,err}`.
- Code: each agent's clone on the laptop; posted commits on branch `agent/<a>` of `~/boltz2-swarm/repo.git`.
- `docs/swarm_log.md`: decision and event history.

"""Laptop-side interface to Polaris for one swarm agent, built on the ALCF IRI API and Globus Transfer (no ssh).

The repo, the board and all development live on this laptop. Polaris only runs PBS jobs, one at a time per agent:
  run NAME --minutes M 'COMMAND'   archive your committed HEAD, upload it to $REMOTE/agents/<you>/code/<sha>/, submit a 1-node 4xA100 job
                                   that unpacks it into a fresh workdir and runs COMMAND there (Boltz venv active); prints the job id
  status [ID]                      state of your current (or given) job; refuses nothing, just reports
  wait ID [--every S] [--max S]    poll until the job ends, printing state and the log tail every S seconds
  log ID [--lines N]               tail of the job's stdout/stderr
  cancel ID                        cancel a job
  ls PATH / cat PATH [--lines N]   list or read a path under $REMOTE/agents/<you>/ (or an absolute /eagle path)
  get PATH LOCAL                   copy a file or directory (recursive) from Polaris to the laptop via Globus
  post-result NAME                 download the run's validation tables and post them to the local board (scoring code is board.py)
  board ARGS...                    experiments/board.py on the local board (leaderboard | read | say | agents)

Rules enforced here: one active job per agent (queued or running), --minutes at most MAX_MINUTES, and tracked files committed.
Env: SWARM_AGENT (required), SWARM_ROOT (local, default ~/boltz2-swarm), IRI_REPO (gm_hackathon_agentic_ai clone with mcp/ and .env),
REMOTE_ROOT (default /eagle/GenesisHackathonOct26/hkwang/boltz2-swarm), VENV_DIR (Polaris, default /home/hkwang/venvs/boltz2).
"""

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

IRI_REPO = Path(os.environ.get("IRI_REPO", Path.home() / "Documents/codex/myproject/gm_hackathon_agentic_ai"))
sys.path.insert(0, str(IRI_REPO / "mcp" / "hpc"))
sys.path.insert(0, str(IRI_REPO / "mcp"))
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
from alcf_iri_client import ALCFIRIClient  # noqa: E402
from auth_env import read_token, read_value  # noqa: E402

POLARIS = "55c1c993-1124-47f9-b823-514ba3849a9a"
EAGLE = "1c3ad9d4-2e91-42bc-becb-72b1fde1235c"
EAGLE_GLOBUS = "05d2c76a-e867-4f67-aa57-76edeb0beda0"
ACCOUNT, QUEUE = "GenesisHackathonOct26", "R7728155"
MAX_MINUTES = int(os.environ.get("SWARM_MAX_MINUTES", "480"))
REMOTE = os.environ.get("REMOTE_ROOT", "/eagle/GenesisHackathonOct26/hkwang/boltz2-swarm")
VENV = os.environ.get("VENV_DIR", "/home/hkwang/venvs/boltz2")
LOCAL = Path(os.environ.get("SWARM_ROOT", Path.home() / "boltz2-swarm"))
ACTIVE = {"queued", "active", "running", "held", "new"}

client = ALCFIRIClient()
client.token_provider = lambda: read_token("alcf")
client.transfer_token_provider = lambda: read_value("GLOBUS_TRANSFER_TOKEN")


def paced(method):
    """IRI allows 5 requests/s per user, shared by every agent on this ALCF account. Space each process's calls by
    MIN_INTERVAL and retry 429 and transient 5xx/connection errors with jittered exponential backoff (about 4 min in total)."""
    async def call(*args, **kwargs):
        for attempt in range(9):
            wait = MIN_INTERVAL - (time.time() - paced.last)
            if wait > 0:
                await asyncio.sleep(wait)
            paced.last = time.time()
            try:
                return await method(*args, **kwargs)
            except Exception as exc:
                text = str(exc)
                transient = any(code in text for code in ("HTTP 429", "429 Too Many", "HTTP 502", "HTTP 503", "HTTP 504", " 502 ", " 503 ", " 504 ")) \
                    or type(exc).__name__ in ("ConnectError", "ReadTimeout", "ConnectTimeout", "RemoteProtocolError")
                if not transient or attempt == 8:
                    raise
                await asyncio.sleep(min(60.0, 2 ** attempt) * (0.5 + random.random()))
    return call


paced.last = 0.0
MIN_INTERVAL = float(os.environ.get("IRI_MIN_INTERVAL", "1.0"))
for _name in ("get_job_status", "submit_job", "cancel_job", "fs_ls", "fs_tail", "fs_view", "get_task", "globus_transfer", "globus_mkdir", "globus_get_task"):
    setattr(client, _name, paced(getattr(client, _name)))


def agent() -> str:
    name = os.environ.get("SWARM_AGENT", "")
    if not name or not name.replace("-", "").replace("_", "").isalnum():
        raise SystemExit("set SWARM_AGENT to your agent name (letters, digits, - and _)")
    return name


def agent_remote() -> str:
    return f"{REMOTE}/agents/{agent()}"


def jobs_file() -> Path:
    path = LOCAL / "agents" / agent() / "jobs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def job_records() -> list[dict]:
    path = jobs_file()
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def find_job(job_id: str) -> dict:
    for record in job_records():
        if record["job_id"] == job_id or record["job_id"].split(".")[0] == job_id or record["name"] == job_id:
            return record
    raise SystemExit(f"no job {job_id} in {jobs_file()}")


def globus_path(path: str) -> str:
    return path[len("/eagle"):] if path.startswith("/eagle/") else path


def resolve(path: str) -> str:
    return path if path.startswith("/") else f"{agent_remote()}/{path}"


async def transfer(source_ep: str, source: str, dest_ep: str, dest: str, recursive: bool, label: str) -> None:
    task = await client.globus_transfer(source_ep, source, dest_ep, dest, label=label, recursive=recursive)
    result = await client.globus_wait_for_transfer(task["task_id"], max_wait=1800)
    if result.get("status") != "SUCCEEDED":
        raise SystemExit(f"Globus transfer {task['task_id']} {result.get('status')}: {result.get('nice_status_details')}")


async def job_state(job_id: str) -> tuple[str, object]:
    status = await client.get_job_status(POLARIS, job_id)
    state = status.get("status", {}) if isinstance(status, dict) else {}
    return str(state.get("state", "unknown")).lower(), state.get("exit_code")


NOISE = ("Warning", "warn(", "warn_eager_only", "warnings.warn", "deprecated", "return func(", "Predicting:", "it/s]", "does not have many workers",
         "LOCAL_RANK", "Initializing distributed", "Tensor Cores", "GPU available", "TPU available", "HPU available", "-----", "distributed_backend", "processes registered", "Consider increasing")


def useful(text: str, lines: int) -> str:
    """Drop library warnings and progress-bar redraws so the training steps and errors are what an agent sees."""
    kept = []
    for line in text.splitlines():
        if (line.strip() and not any(marker in line for marker in NOISE) or "Error" in line) and (not kept or line != kept[-1]):
            kept.append(line)
    return "\n".join(line[:400] for line in kept[-lines:])


async def tail(path: str, lines: int, raw: bool = False) -> str:
    try:
        result = await client.fs_tail(EAGLE, path, lines if raw else max(400, lines * 20))
        if isinstance(result, dict) and "task_id" in result:
            result = (await client.wait_for_task(result["task_id"])).get("result", {})
        output = result.get("output", result) if isinstance(result, dict) else result
        if isinstance(output, dict) and "content" in output:
            return str(output["content"]) if raw else useful(str(output["content"]), lines)
        if isinstance(result, dict) and result.get("error"):
            return f"({result['error']})"
        return str(output)
    except Exception as exc:
        return f"(could not read {path}: {exc})"


async def cmd_run(args) -> None:
    for record in job_records()[::-1]:
        if record.get("final_state"):
            continue
        state, _ = await job_state(record["job_id"])
        if state in ACTIVE:
            raise SystemExit(f"you already have job {record['job_id']} ({record['name']}, {state}); one job at a time. wait or cancel it first")
        await record_final(record, state)
    if not 1 <= args.minutes <= MAX_MINUTES:
        raise SystemExit(f"--minutes must be 1..{MAX_MINUTES}")
    if not args.name.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise SystemExit("bad NAME (letters, digits, . _ -)")
    if any(record["name"] == args.name for record in job_records()):
        raise SystemExit(f"job name {args.name} already used; pick a new one (the candidate NAME inside COMMAND may repeat to resume training)")
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip():
        raise SystemExit("commit your tracked changes first: the job runs your committed HEAD")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    code_dir = f"{agent_remote()}/code/{sha[:12]}"
    work = f"{agent_remote()}/work/{args.name}"
    stage = Path(tempfile.mkdtemp(prefix=f"swarm-{agent()}-", dir=Path.home() / "tmp"))
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(stage / "code.tar.gz"), "HEAD", "--", ".", ":(exclude)data/raw"], check=True)
    command = args.command
    (stage / f"job_{args.name}.sh").write_text(f"""#!/bin/bash -l
set -eo pipefail
export AGENT={agent()} NAME={args.name} COMMIT={sha} SWARM_REMOTE={REMOTE}
work={work}
exec >>{agent_remote()}/jobs/{args.name}.log 2>&1
rm -rf "$work" && mkdir -p "$work" && cd "$work"
tar -xzf {code_dir}/code.tar.gz
ln -sfn {REMOTE}/shared/structures data/structures
ln -sfn {REMOTE}/shared/boltz_inputs data/processed/boltz_inputs
mkdir -p {agent_remote()}/results && ln -sfn {agent_remote()}/results experiments/results
module load cray-python/3.12.12
module load cuda/12.9
source {VENV}/bin/activate
export CC=gcc PYTHONPATH="$work/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 TMPDIR=/tmp
echo "== $(date -u +%FT%TZ) $(hostname) job $PBS_JOBID agent $AGENT name $NAME commit $COMMIT"
nvidia-smi -L || true
{command}
""")
    await client.globus_mkdir(EAGLE_GLOBUS, globus_path(code_dir))
    await client.globus_mkdir(EAGLE_GLOBUS, globus_path(f"{agent_remote()}/jobs"))
    await transfer(client._local_endpoint(), str(stage), EAGLE_GLOBUS, globus_path(code_dir) + "/", True, f"{agent()} {args.name} code")
    spec = {
        "executable": "/bin/bash", "arguments": ["-l", f"{code_dir}/job_{args.name}.sh"], "name": args.name[:15],
        "resources": {"node_count": 1},
        "attributes": {"duration": args.minutes * 60, "queue_name": QUEUE, "account": ACCOUNT, "custom_attributes": {"filesystems": "home:eagle"}},
        "stdout_path": f"{agent_remote()}/jobs/{args.name}.out", "stderr_path": f"{agent_remote()}/jobs/{args.name}.err",
    }
    result = await client.submit_job(POLARIS, spec)
    job_id = result.get("id") or result.get("job_id")
    record = {"job_id": job_id, "name": args.name, "commit": sha, "minutes": args.minutes, "submitted": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "command": command, "stdout": f"{agent_remote()}/jobs/{args.name}.log", "stderr": spec["stderr_path"], "workdir": work}
    with open(jobs_file(), "a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(f"submitted {job_id} ({args.name}, {args.minutes} min, commit {sha[:7]}); workdir {work}")


async def record_final(record: dict, state: str, code=None) -> None:
    if state in ACTIVE or state == "unknown":
        return
    records = job_records()
    for other in records:
        if other["job_id"] == record["job_id"]:
            other["final_state"], other["exit_code"] = state, code
    jobs_file().write_text("".join(json.dumps(other) + "\n" for other in records))


async def cmd_status(args) -> None:
    records = [find_job(args.id)] if args.id else job_records()[-8:]
    if not records:
        print("no jobs yet")
    for record in records:
        if record.get("final_state"):
            state, code = record["final_state"], record.get("exit_code")
        else:
            state, code = await job_state(record["job_id"])
            await record_final(record, state, code)
        print(f"{record['job_id']}  {record['name']:28} {state:10} exit {code}  {record['minutes']} min  commit {record['commit'][:7]}  submitted {record['submitted']}")


async def cmd_wait(args) -> None:
    record = find_job(args.id)
    started = time.time()
    while True:
        state, code = await job_state(record["job_id"])
        print(f"== {time.strftime('%H:%M:%S')} {record['name']} {state} exit {code}", flush=True)
        if state not in ACTIVE:
            await record_final(record, state, code)
            print((await tail(record["stdout"], args.lines)).rstrip())
            err = (await tail(record["stderr"], 15)).rstrip()
            if err and "does not exist" not in err:
                print("-- stderr\n" + err)
            return
        if state != "queued":
            print((await tail(record["stdout"], 5)).rstrip(), flush=True)
        if time.time() - started > args.max:
            print(f"still {state} after {args.max}s; call wait again")
            return
        await asyncio.sleep(args.every)


async def cmd_log(args) -> None:
    record = find_job(args.id)
    print((await tail(record["stdout"], args.lines, args.raw)).rstrip())
    err = (await tail(record["stderr"], min(args.lines, 40), args.raw)).rstrip()
    if err and "does not exist" not in err:
        print("-- PBS stderr (written when the job ends)\n" + err)


async def cmd_cancel(args) -> None:
    record = find_job(args.id)
    await client.cancel_job(POLARIS, record["job_id"])
    print(f"cancel requested for {record['job_id']}")


async def cmd_ls(args) -> None:
    result = await client.fs_ls(EAGLE, resolve(args.path))
    if isinstance(result, dict) and "task_id" in result:
        result = (await client.wait_for_task(result["task_id"])).get("result", {})
    for entry in result.get("output", []) if isinstance(result, dict) else []:
        print(f"{entry['type'][0]} {entry['size']:>12} {entry['last_modified']}  {entry['name']}")
    if isinstance(result, dict) and result.get("error"):
        print(result["error"])


async def cmd_cat(args) -> None:
    print((await tail(resolve(args.path), args.lines, raw=True)).rstrip())


async def cmd_get(args) -> None:
    source = resolve(args.path)
    local = Path(args.local).expanduser().resolve()
    listing = await client.fs_ls(EAGLE, source)
    if isinstance(listing, dict) and "task_id" in listing:
        listing = (await client.wait_for_task(listing["task_id"])).get("result", {})
    entries = listing.get("output", []) if isinstance(listing, dict) else []
    is_dir = not (len(entries) == 1 and entries[0]["type"] == "file" and source.endswith(entries[0]["name"]))
    local.parent.mkdir(parents=True, exist_ok=True)
    await transfer(EAGLE_GLOBUS, globus_path(source) + ("/" if is_dir else ""), client._local_endpoint(), str(local) + ("/" if is_dir else ""),
                   is_dir, f"{agent()} get {Path(source).name}")
    print(f"copied {source} -> {local}")


async def cmd_post_result(args) -> None:
    """Fetch val_<NAME>_seed<S>/eval_per_entry.csv for each seed and score them with board.py post-result on the local board."""
    record = find_job(args.job) if args.job else None
    label = args.label
    root = LOCAL / "agents" / agent() / "evals" / label
    eval_specs = []
    for seed in args.seeds.split():
        remote = f"{agent_remote()}/results/mac1_eval/val_{label}_seed{seed}/eval_per_entry.csv"
        local = root / f"seed{seed}" / "eval_per_entry.csv"
        local.parent.mkdir(parents=True, exist_ok=True)
        await transfer(EAGLE_GLOBUS, globus_path(remote), client._local_endpoint(), str(local), False, f"{agent()} eval {label} {seed}")
        eval_specs.append(f"{seed}:{local.parent}")
    commit = record["commit"] if record else args.commit
    if not commit:
        raise SystemExit("pass --job JOB_ID (or --commit SHA) so the result is recorded with the code that produced it")
    board(["post-result", "--agent", agent(), "--label", label, "--method", args.method, "--notes", args.notes,
           "--ckpt", f"{agent_remote()}/results/runs/{label}/boltz_ckpt/final", "--train-run", f"{agent_remote()}/results/runs/{label}",
           "--commit", commit, "--job", record["job_id"] if record else "", "--eval", *eval_specs])


def board(argv: list[str]) -> None:
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "SWARM_ROOT": str(LOCAL)}
    sys.exit(subprocess.run([sys.executable, str(repo / "experiments" / "board.py"), *argv], env=env).returncode)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "board":
        board(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("name")
    run.add_argument("command")
    run.add_argument("--minutes", type=int, required=True)
    run.set_defaults(run=cmd_run)
    status = commands.add_parser("status")
    status.add_argument("id", nargs="?")
    status.set_defaults(run=cmd_status)
    wait = commands.add_parser("wait")
    wait.add_argument("id")
    wait.add_argument("--every", type=int, default=60)
    wait.add_argument("--max", type=int, default=1800)
    wait.add_argument("--lines", type=int, default=30)
    wait.set_defaults(run=cmd_wait)
    log = commands.add_parser("log")
    log.add_argument("id")
    log.add_argument("--lines", type=int, default=60)
    log.add_argument("--raw", action="store_true", help="do not filter warnings and progress bars")
    log.set_defaults(run=cmd_log)
    cancel = commands.add_parser("cancel")
    cancel.add_argument("id")
    cancel.set_defaults(run=cmd_cancel)
    ls = commands.add_parser("ls")
    ls.add_argument("path", nargs="?", default="")
    ls.set_defaults(run=cmd_ls)
    cat = commands.add_parser("cat")
    cat.add_argument("path")
    cat.add_argument("--lines", type=int, default=60)
    cat.set_defaults(run=cmd_cat)
    get = commands.add_parser("get")
    get.add_argument("path")
    get.add_argument("local")
    get.set_defaults(run=cmd_get)
    post = commands.add_parser("post-result")
    post.add_argument("label")
    post.add_argument("--method", required=True)
    post.add_argument("--notes", default="")
    post.add_argument("--job", default="")
    post.add_argument("--commit", default="")
    post.add_argument("--seeds", default="2026 1 2")
    post.set_defaults(run=cmd_post_result)
    args = parser.parse_args()
    if args.command == "ls" and not args.path:
        args.path = agent_remote()
    asyncio.run(args.run(args))


if __name__ == "__main__":
    main()

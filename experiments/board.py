"""Shared results board and message channel for fine-tuning agents that run in separate PBS jobs.

Everything lives in $SWARM_ROOT/board (default /eagle/GenesisHackathonOct26/hkwang/boltz2-swarm/board) as append-only JSON lines
written under a file lock, so agents on different nodes can use it at the same time:
  results.jsonl   one scored candidate per line (written only by `post-result`)
  messages.jsonl  notes, claims, questions and warnings between agents
  agents.jsonl    agent registrations
  frozen.json     sha256 of the files that define the split and the scoring, plus the validation entry ids
  baseline/       per-seed stock (pretrained) validation tables, validation rows only

Run `python experiments/board.py --help` for the commands. The score is the validation split only: `split == "val"` rows are
selected before anything else is read, so test rows never enter a metric.
"""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SWARM_ROOT = Path(os.environ.get("SWARM_ROOT", "/eagle/GenesisHackathonOct26/hkwang/boltz2-swarm"))
BOARD = SWARM_ROOT / "board"
FROZEN_FILES = [
    "data/processed/mac1_splits.csv",
    "data/processed/mac1_training_records.csv",
    "data/processed/mac1_eval/eval_entries.csv",
    "experiments/evaluate_mac1.py",
    "src/boltz2_finetuning/evaluation.py",
    "configs/mac1_cofold.yaml",
    "experiments/run_mac1_eval.sh",
    "experiments/run_val_eval.sh",
    "experiments/board.py",
]
FROZEN_DIRS = ["data/processed/mac1_eval/inputs"]
BOOTSTRAPS = 2000


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def append(name: str, prefix: str, record: dict) -> dict:
    """Append one record to a board file under an exclusive lock; adds a sequential `id` and a `time`."""
    with open(BOARD / name, "a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        record = {"id": f"{prefix}{sum(1 for _ in handle) + 1}", "time": now(), **record}
        handle.write(json.dumps(record) + "\n")
        handle.flush()
    return record


def read_all(name: str) -> list[dict]:
    path = BOARD / name
    if not path.exists():
        return []
    with open(path) as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        return [json.loads(line) for line in handle if line.strip()]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_hashes(repo: Path) -> dict[str, str]:
    hashes = {name: file_hash(repo / name) for name in FROZEN_FILES}
    for directory in FROZEN_DIRS:
        for path in sorted((repo / directory).glob("*")):
            hashes[str(path.relative_to(repo))] = file_hash(path)
    return hashes


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def val_table(path: Path) -> pd.DataFrame:
    """Validation rows of an eval_per_entry.csv, indexed by pdb_id. Rows of any other split are dropped here."""
    table = pd.read_csv(path)
    return table[table["split"] == "val"].set_index("pdb_id")


def score(tables: dict[int, pd.DataFrame]) -> dict:
    """Validation metrics from {predict seed: val table}. Primary: mean over seeds of the median top-1 ligand RMSD (A, lower is better).

    `delta_top1` is the median over entries of the change in top-1 RMSD against the stock model, each entry averaged over the
    seeds that have a stock baseline, with a bootstrap 95% interval over entries.
    """
    val_ids = sorted(json.loads((BOARD / "frozen.json").read_text())["val_ids"])
    per_seed, entry_deltas = {}, []
    for seed, table in tables.items():
        if set(table.index) != set(val_ids) or table["rmsd_top1"].isna().any():
            raise SystemExit(f"seed {seed}: evaluation incomplete ({len(table)} val rows, {int(table['rmsd_top1'].isna().sum())} without a pose)")
        measured = table[(table["ic50_qualifier"] == "=") & table["affinity_pred_value"].notna()]
        per_seed[seed] = {
            "median_top1": float(table["rmsd_top1"].median()),
            "median_best": float(table["rmsd_best"].median()),
            "frac_below_2A": float((table["rmsd_top1"] < 2.0).mean()),
            "spearman": float(stats.spearmanr(measured["affinity_pred_value"], measured["log10_ic50_uM"]).statistic) if len(measured) > 2 else None,
        }
        baseline_path = BOARD / "baseline" / f"stock_seed{seed}.csv"
        if baseline_path.exists():
            baseline = pd.read_csv(baseline_path).set_index("pdb_id")
            entry_deltas.append((table["rmsd_top1"] - baseline["rmsd_top1"]).reindex(val_ids))
    spearmans = [metrics["spearman"] for metrics in per_seed.values() if metrics["spearman"] is not None]
    result = {
        "seeds": list(per_seed),
        "val_rmsd_top1": float(np.mean([metrics["median_top1"] for metrics in per_seed.values()])),
        "val_rmsd_best": float(np.mean([metrics["median_best"] for metrics in per_seed.values()])),
        "val_frac_below_2A": float(np.mean([metrics["frac_below_2A"] for metrics in per_seed.values()])),
        "val_spearman": float(np.mean(spearmans)) if spearmans else None,
        "seed_spread_top1": float(np.ptp([metrics["median_top1"] for metrics in per_seed.values()])),
        "per_seed": per_seed,
        "delta_top1": None,
        "delta_top1_ci95": None,
    }
    if entry_deltas:
        delta = np.mean(entry_deltas, axis=0)
        generator = np.random.default_rng(0)
        medians = [np.median(generator.choice(delta, len(delta))) for _ in range(BOOTSTRAPS)]
        result.update(delta_top1=float(np.median(delta)), delta_top1_ci95=[float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))])
    return result


def command_init(args) -> None:
    for name in ("baseline", "cursors"):
        (BOARD / name).mkdir(parents=True, exist_ok=True)
    for name in ("results.jsonl", "messages.jsonl", "agents.jsonl"):
        (BOARD / name).touch()
    entries = pd.read_csv(args.repo / "data/processed/mac1_eval/eval_entries.csv")
    frozen = {"created": now(), "commit": git(args.repo, "rev-parse", "HEAD"), "val_ids": sorted(entries.loc[entries["split"] == "val", "pdb_id"]),
              "files": frozen_hashes(args.repo)}
    (BOARD / "frozen.json").write_text(json.dumps(frozen, indent=2))
    tables = {}
    for spec in args.stock:
        seed, directory = spec.split(":", 1)
        table = val_table(Path(directory) / "eval_per_entry.csv")
        table[["split", "rmsd_top1", "rmsd_best", "affinity_pred_value", "ic50_qualifier", "log10_ic50_uM"]].to_csv(BOARD / "baseline" / f"stock_seed{seed}.csv")
        tables[int(seed)] = table
    result = score(tables)
    append("results.jsonl", "r", {"agent": "operator", "label": "stock", "method": "pretrained Boltz-2, no fine-tuning", "branch": None,
                                 "commit": frozen["commit"], **result})
    print(f"board initialised at {BOARD}: {len(frozen['val_ids'])} val entries, stock val_rmsd_top1 {result['val_rmsd_top1']:.3f}")


def command_check_frozen(args) -> None:
    frozen = json.loads((BOARD / "frozen.json").read_text())["files"]
    current = frozen_hashes(args.repo)
    changed = sorted({name for name in set(frozen) | set(current) if frozen.get(name) != current.get(name)})
    if changed:
        raise SystemExit("frozen files differ from the board's copy (split and scoring code must not change): " + ", ".join(changed))
    print("frozen split and scoring files unchanged")


def command_refreeze(args) -> None:
    """Operator: record the current hashes of the frozen files (after an intentional change to the evaluation)."""
    frozen = json.loads((BOARD / "frozen.json").read_text())
    frozen.update(refrozen=now(), commit=git(args.repo, "rev-parse", "HEAD"), files=frozen_hashes(args.repo))
    (BOARD / "frozen.json").write_text(json.dumps(frozen, indent=2))
    append("messages.jsonl", "m", {"agent": "operator", "kind": "warning", "reply_to": None, "text": f"frozen evaluation files were updated at {frozen['commit'][:7]}; rebase onto it before posting results"})
    print("frozen hashes updated")


def command_register(args) -> None:
    record = append("agents.jsonl", "a", {"agent": args.agent, "job": os.environ.get("PBS_JOBID"), "host": os.uname().nodename,
                                          "brief": args.brief, "deadline": args.deadline})
    print(f"registered {record['agent']} ({record['id']})")


def command_say(args) -> None:
    record = append("messages.jsonl", "m", {"agent": args.agent, "kind": args.kind, "reply_to": args.reply_to, "text": " ".join(args.text)})
    print(f"posted {record['id']}")


def command_read(args) -> None:
    messages = read_all("messages.jsonl")
    cursor = BOARD / "cursors" / (args.agent or "anonymous")
    seen = int(cursor.read_text()) if args.unread and cursor.exists() else 0
    shown = [message for message in messages if int(message["id"][1:]) > seen]
    shown = shown[-args.tail:] if args.tail else shown
    for message in shown:
        reply = f" re {message['reply_to']}" if message.get("reply_to") else ""
        print(f"[{message['id']} {message['time']} {message['agent']} {message['kind']}{reply}] {message['text']}")
    if not shown:
        print("no new messages" if args.unread else "no messages")
    if args.unread and args.agent and messages:
        cursor.write_text(messages[-1]["id"][1:])


def command_post_result(args) -> None:
    repo = Path.cwd()
    dirty = bool(git(repo, "status", "--porcelain", "--untracked-files=no"))
    if dirty and not args.allow_dirty:
        raise SystemExit("tracked files have uncommitted changes: commit them first so the result can be reproduced (or pass --allow-dirty)")
    command_check_frozen(argparse.Namespace(repo=repo))
    tables = {}
    for spec in args.eval:
        seed, directory = spec.split(":", 1)
        tables[int(seed)] = val_table(Path(directory) / "eval_per_entry.csv")
    result = score(tables)
    record = append("results.jsonl", "r", {
        "agent": args.agent, "label": args.label, "method": args.method, "notes": args.notes, "job": os.environ.get("PBS_JOBID"),
        "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD"), "commit": git(repo, "rev-parse", "HEAD"), "dirty": dirty,
        "ckpt": args.ckpt, "train_run": args.train_run, "eval_dirs": [spec.split(":", 1)[1] for spec in args.eval], **result})
    delta = "" if result["delta_top1"] is None else f", delta vs stock {result['delta_top1']:+.3f} A [{result['delta_top1_ci95'][0]:+.3f}, {result['delta_top1_ci95'][1]:+.3f}]"
    text = f"{record['id']} {args.label}: val_rmsd_top1 {result['val_rmsd_top1']:.3f} A over seeds {result['seeds']}{delta}; {args.method}"
    append("messages.jsonl", "m", {"agent": args.agent, "kind": "result", "reply_to": None, "text": text})
    print(text)
    if not dirty:
        pushed = subprocess.run(["git", "-C", str(repo), "push", "-q", "origin", f"HEAD:refs/heads/{record['branch']}"], capture_output=True, text=True)
        print("pushed code to the hub" if pushed.returncode == 0 else f"warning: could not push to the hub: {pushed.stderr.strip()}")


def command_leaderboard(args) -> None:
    rows = sorted(read_all("results.jsonl"), key=lambda row: row["val_rmsd_top1"])
    stock = next(row["val_rmsd_top1"] for row in rows if row["label"] == "stock")
    print(f"{'rank':>4} {'id':>4} {'agent':8} {'label':28} {'seeds':>5} {'top1':>6} {'delta [95% CI]':>22} {'<2A':>5} {'spear':>5} {'spread':>6}  branch@commit  method")
    for rank, row in enumerate(rows[: args.top], 1):
        delta = "" if row["delta_top1"] is None else "%+.3f [%+.3f,%+.3f]" % (row["delta_top1"], *row["delta_top1_ci95"])
        where = f"{row['branch']}@{row['commit'][:7]}" + ("*" if row.get("dirty") else "")
        spearman = "" if row["val_spearman"] is None else f"{row['val_spearman']:.2f}"
        print(f"{rank:>4} {row['id']:>4} {row['agent'][:8]:8} {row['label'][:28]:28} {len(row['seeds']):>5} {row['val_rmsd_top1']:6.3f} {delta:>22} "
              f"{row['val_frac_below_2A']:5.2f} {spearman:>5} {row['seed_spread_top1']:6.2f}  {where}  {row['method'][:60]}")
    print(f"stock = {stock:.3f}; 'spread' is the seed-to-seed range of one model's median RMSD (sampling noise); * = uncommitted code")


def command_agents(args) -> None:
    latest = {record["agent"]: record for record in read_all("agents.jsonl")}
    for record in latest.values():
        print(f"{record['agent']:10} job {record['job']} host {record['host']} started {record['time']} deadline {record['deadline']} brief {record['brief']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="operator: create the board from a reference checkout and stock evaluations")
    init.add_argument("--repo", type=Path, required=True)
    init.add_argument("--stock", nargs="+", required=True, help="SEED:EVAL_DIR for each stock evaluation")
    init.set_defaults(run=command_init)

    check = commands.add_parser("check-frozen", help="fail if the split or scoring files differ from the board's copy")
    check.add_argument("--repo", type=Path, default=Path.cwd())
    check.set_defaults(run=command_check_frozen)

    refreeze = commands.add_parser("refreeze", help="operator: accept the current evaluation files as the frozen ones")
    refreeze.add_argument("--repo", type=Path, default=Path.cwd())
    refreeze.set_defaults(run=command_refreeze)

    register = commands.add_parser("register")
    register.add_argument("--agent", required=True)
    register.add_argument("--brief", default="")
    register.add_argument("--deadline", default="")
    register.set_defaults(run=command_register)

    say = commands.add_parser("say", help="post a message to the other agents")
    say.add_argument("--agent", required=True)
    say.add_argument("--kind", choices=["note", "claim", "question", "warning", "idea"], default="note")
    say.add_argument("--reply-to")
    say.add_argument("text", nargs="+")
    say.set_defaults(run=command_say)

    read = commands.add_parser("read", help="print messages (--unread: only new ones for --agent, then advance its cursor)")
    read.add_argument("--agent")
    read.add_argument("--unread", action="store_true")
    read.add_argument("--tail", type=int, default=0)
    read.set_defaults(run=command_read)

    post = commands.add_parser("post-result", help="score validation evaluations and add them to the leaderboard")
    post.add_argument("--agent", required=True)
    post.add_argument("--label", required=True)
    post.add_argument("--method", required=True)
    post.add_argument("--eval", nargs="+", required=True, help="SEED:EVAL_DIR for each predict seed")
    post.add_argument("--ckpt", default=None)
    post.add_argument("--train-run", dest="train_run", default=None)
    post.add_argument("--notes", default="")
    post.add_argument("--allow-dirty", action="store_true")
    post.set_defaults(run=command_post_result)

    leaderboard = commands.add_parser("leaderboard")
    leaderboard.add_argument("--top", type=int, default=20)
    leaderboard.set_defaults(run=command_leaderboard)

    commands.add_parser("agents", help="list registered agents").set_defaults(run=command_agents)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    sys.exit(main())

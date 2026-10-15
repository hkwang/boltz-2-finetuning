"""Summarize gpu_monitor.sh logs: per-GPU load, and which GPUs ran a process.

Usage: python experiments/summarize_gpu_monitor.py PREFIX [--idle-pct 10]
PREFIX is the gpu_monitor.sh output prefix (reads PREFIX.gpu.csv, PREFIX.apps.csv).
Exits 1 if any GPU that had a process averaged below --idle-pct utilization,
or if a GPU that ran no process exists while another GPU did.
"""
import argparse
import csv
from collections import defaultdict
from statistics import mean


def read_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle, skipinitialspace=True))
    return [{key.strip(): value.strip() for key, value in row.items()} for row in rows]


def number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix")
    parser.add_argument("--idle-pct", type=float, default=10.0)
    args = parser.parse_args()

    gpu_rows = read_rows(f"{args.prefix}.gpu.csv")
    app_rows = read_rows(f"{args.prefix}.apps.csv")
    if not gpu_rows:
        print("No GPU samples recorded")
        return 1

    samples = defaultdict(list)
    for row in gpu_rows:
        samples[(row["index"], row["uuid"])].append(row)
    pids_by_uuid = defaultdict(set)
    for row in app_rows:
        pids_by_uuid[row["gpu_uuid"]].add(row["pid"])

    n_samples = len(next(iter(samples.values())))
    print(f"{len(samples)} GPUs, {n_samples} samples per GPU")
    print("gpu  util_mean util_max  below_idle  mem_max_MiB  power_mean_W  pids")
    problems = []
    gpus_with_process = [key for key in samples if pids_by_uuid[key[1]]]
    for (index, uuid), rows in sorted(samples.items(), key=lambda item: int(item[0][0])):
        util = [v for v in (number(r["utilization.gpu"]) for r in rows) if v is not None]
        mem = [v for v in (number(r["memory.used"]) for r in rows) if v is not None]
        power = [v for v in (number(r["power.draw"]) for r in rows) if v is not None]
        below = sum(v < args.idle_pct for v in util) / len(util)
        pids = pids_by_uuid[uuid]
        print(
            f"{index:>3}  {mean(util):9.1f} {max(util):8.0f}  {below:10.0%}"
            f"  {max(mem):11.0f}  {mean(power):12.0f}  {','.join(sorted(pids)) or '-'}"
        )
        if pids and mean(util) < args.idle_pct:
            problems.append(f"GPU {index} had a process but averaged {mean(util):.1f}% utilization")
        if not pids and gpus_with_process:
            problems.append(f"GPU {index} ran no process while other GPUs did")

    for problem in problems:
        print(f"WARNING: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Results-at-a-glance figures for docs/report.md: pose (best-of-5 ligand RMSD) and affinity (Spearman, RMSE by split).

Run: python experiments/plot_results_glance.py [--out-dir docs/figures]
Writes two figures with the same layout and different error bars:
  results_at_a_glance_seeds.png      mean over prediction seeds (2026, 1, 2) with a 95% t-interval over seeds (n = 3); small dots are the seeds.
  results_at_a_glance_bootstrap.png  one marker per seed, each with a 95% bootstrap interval over held-out entries (2000 resamples).
Reads experiments/results/mac1_eval/<run>/eval_per_entry.csv (pose; stock-head affinity on fine-tuned seed-2026 poses),
experiments/results/finetune_affinity/<head>/eval_step500.csv (tuned heads on fine-tuned seed-2026 poses),
experiments/results/finetune_affinity/stockpose/<head>.csv (all heads on stock seed-2026 poses) and
experiments/results/finetune_affinity/heads_on_poses/<run>/<head>.csv (all heads on seed-1 and seed-2 poses).
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVAL = Path("experiments/results/mac1_eval")
AFFINITY = Path("experiments/results/finetune_affinity")
BOOTSTRAPS = 2000
SEEDS = (2026, 1, 2)
T_95 = {2: 12.706, 3: 4.303}
SEED_MARKERS = {2026: "o", 1: "s", 2: "^"}
POSE_RUNS = [
    ("stock", "stock"),
    ("full 3e-6, s1400", "full_lr3e-6_step1400"),
    ("full 1e-5, s1400", "full_lr1e-5_step1400"),
    ("full 1e-4, s1400", "full_lr1e-4_step1400"),
    ("full 3e-4, s1000", "full_lr3e-4_step1000"),
    ("full 3e-4, s2000", "full_lr3e-4_step2000"),
    ("LoRA 1e-4, s1400", "lora_r16_lr1e-4_step1400"),
    ("LoRA 3e-4, s2000", "lora_r16_lr3e-4_step2000"),
]
HEADS = ["stock", "aff_lr1e-5", "aff_lr2e-5", "aff_lr5e-5", "aff_nopair"]
POSES = {"stock pose": "stock", "fine-tuned pose": "full_lr1e-4_step1400"}
POSE_COLORS = {"stock pose": "#4c72b0", "fine-tuned pose": "#dd8452"}


def seed_run(run: str, seed: int) -> str:
    return run if seed == 2026 else f"{run}_seed{seed}"


def by_pdb(table: pd.DataFrame, column: str) -> pd.Series:
    return table.set_index(table["record_id"].str.replace("mac1_eval_", ""))[column]


def spearman(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.corrcoef(pd.Series(pred).rank(), pd.Series(truth).rank())[0, 1])


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(((pred - truth) ** 2).mean()))


def bootstrap(statistic, arrays: tuple[np.ndarray, ...], rng: np.random.Generator) -> tuple[float, float]:
    size = len(arrays[0])
    draws = []
    for _ in range(BOOTSTRAPS):
        sample = rng.integers(0, size, size)
        draws.append(statistic(*(array[sample] for array in arrays)))
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return float(low), float(high)


def seed_interval(values: list[float]) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean, mean
    half = T_95[len(values)] * float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return mean, mean - half, mean + half


def head_predictions(pose_run: str, seed: int, head: str) -> pd.Series:
    if seed != 2026:
        return by_pdb(pd.read_csv(AFFINITY / "heads_on_poses" / seed_run(pose_run, seed) / f"{head}.csv"), "pred")
    if pose_run == "stock":
        return by_pdb(pd.read_csv(AFFINITY / "stockpose" / f"{head}.csv"), "pred")
    if head == "stock":
        return pd.read_csv(EVAL / pose_run / "eval_per_entry.csv").set_index("pdb_id")["affinity_pred_value"]
    return by_pdb(pd.read_csv(AFFINITY / head / "eval_step500.csv"), "pred")


def pose_values(rng: np.random.Generator) -> list[dict]:
    rows = []
    for label, run in POSE_RUNS:
        for seed in SEEDS:
            path = EVAL / seed_run(run, seed) / "eval_per_entry.csv"
            if not path.exists():
                continue
            values = pd.read_csv(path)["rmsd_best"].dropna().to_numpy()
            low, high = bootstrap(np.median, (values,), rng)
            rows.append({"label": label, "seed": seed, "value": float(np.median(values)), "low": low, "high": high})
    return rows


def affinity_values(rng: np.random.Generator) -> list[dict]:
    reference = pd.read_csv(EVAL / "stock" / "eval_per_entry.csv").set_index("pdb_id")
    rows = []
    for split in ("val", "test"):
        entries = reference.index[(reference["ic50_qualifier"] == "=") & (reference["split"] == split)]
        truth = reference.loc[entries, "log10_ic50_uM"].to_numpy()
        for pose, pose_run in POSES.items():
            for head in HEADS:
                for seed in SEEDS:
                    pred = head_predictions(pose_run, seed, head).loc[entries].to_numpy()
                    for metric, statistic in (("spearman", spearman), ("rmse", rmse)):
                        low, high = bootstrap(statistic, (pred, truth), rng)
                        rows.append({"split": split, "n": len(entries), "pose": pose, "head": head, "seed": seed, "metric": metric,
                                     "value": statistic(pred, truth), "low": low, "high": high})
    return rows


def draw(axis: plt.Axes, position: float, rows: list[dict], mode: str, color: str, horizontal: bool, spacing: float) -> None:
    """Seeds mode: one bar (mean, t-interval over seeds) plus small seed dots. Bootstrap mode: one marker and bar per seed."""
    def errorbar(center, value, low, high, **style):
        error = [[value - low], [high - value]]
        if horizontal:
            axis.errorbar(value, center, xerr=error, **style)
        else:
            axis.errorbar(center, value, yerr=error, **style)

    if mode == "seeds":
        mean, low, high = seed_interval([row["value"] for row in rows])
        errorbar(position, mean, low, high, fmt="D", color=color, markersize=7, capsize=4, linewidth=2)
        for row in rows:
            point = (row["value"], position) if horizontal else (position, row["value"])
            axis.plot(*point, SEED_MARKERS[row["seed"]], color=color, markersize=4, alpha=0.55, markeredgewidth=0)
    else:
        offsets = {2026: -spacing, 1: 0.0, 2: spacing}
        for row in rows:
            errorbar(position + offsets[row["seed"]], row["value"], row["low"], row["high"], fmt=SEED_MARKERS[row["seed"]], color=color,
                     markersize=6, capsize=2.5, linewidth=1.3)


def plot_pose(axis: plt.Axes, rows: list[dict], mode: str) -> None:
    labels = [label for label, _ in POSE_RUNS]
    stock_mean = np.mean([row["value"] for row in rows if row["label"] == "stock"])
    axis.axvline(stock_mean, color="0.6", linestyle="--", linewidth=1.2, zorder=0)
    for index, label in enumerate(labels):
        draw(axis, index, [row for row in rows if row["label"] == label], mode, "#4c72b0" if label == "stock" else "#55a868", True, 0.22)
    axis.set_yticks(range(len(labels)), [f"{label} ({sum(row['label'] == label for row in rows)} seed{'s' if sum(row['label'] == label for row in rows) > 1 else ''})" for label in labels])
    axis.invert_yaxis()
    axis.set_xlabel("Median best-of-5 ligand RMSD (Å)")
    axis.set_title("Pose: diffusion fine-tuning (45 held-out entries)")
    axis.grid(axis="x", alpha=0.3)


def plot_affinity(axes, rows: list[dict], mode: str) -> None:
    width = 0.2
    for column, split in enumerate(("val", "test")):
        for row_index, metric in enumerate(("spearman", "rmse")):
            axis = axes[row_index][column]
            for pose_index, pose in enumerate(POSES):
                for head_index, head in enumerate(HEADS):
                    selected = [row for row in rows if (row["split"], row["metric"], row["pose"], row["head"]) == (split, metric, pose, head)]
                    draw(axis, head_index + (pose_index - 0.5) * 2 * width, selected, mode, POSE_COLORS[pose], False, 0.07)
            count = next(row["n"] for row in rows if row["split"] == split)
            axis.set_xticks(range(len(HEADS)), ["stock" if head == "stock" else head.replace("aff_", "") for head in HEADS])
            axis.set_xlim(-0.6, len(HEADS) - 0.4)
            axis.grid(axis="y", alpha=0.3)
            axis.set_title(f"{split} ({count} IC50s)")
            axis.set_ylabel("Spearman ρ" if metric == "spearman" else "RMSE (log10 µM)")
            if row_index == 1:
                axis.set_xlabel("Affinity head (step 500)")


def legend_handles(mode: str) -> list:
    handles = [plt.Line2D([], [], color=color, marker="s", linestyle="none", markersize=9, label=pose) for pose, color in POSE_COLORS.items()]
    handles.append(plt.Line2D([], [], color="0.6", linestyle="--", label="stock pose RMSD, mean of seeds"))
    if mode == "seeds":
        handles.append(plt.Line2D([], [], color="0.3", marker="D", linestyle="-", markersize=7, label="mean of seeds, 95% CI over seeds (t, n = 3)"))
    for seed, marker in SEED_MARKERS.items():
        size = 5 if mode == "seeds" else 7
        handles.append(plt.Line2D([], [], color="0.3", marker=marker, linestyle="none", markersize=size, label=f"seed {seed}"))
    return handles


def make_figure(pose_rows: list[dict], affinity_rows: list[dict], mode: str, path: Path) -> None:
    figure = plt.figure(figsize=(15, 8.2), layout="constrained")
    if mode == "seeds":
        figure.suptitle("Error bars: 95% CI over prediction seeds (2026, 1, 2), not over entries; small dots are the individual seeds",
                        fontsize=13, color="0.25")
    else:
        figure.suptitle("Error bars: 95% bootstrap CI over held-out entries (2000 resamples), one bar per prediction seed",
                        fontsize=13, color="0.25")
    left, right = figure.subfigures(1, 2, width_ratios=[1, 1.35])
    plot_pose(left.subplots(), pose_rows, mode)
    right.suptitle("Affinity: head fine-tuning on stock vs fine-tuned (full 1e-4, s1400) poses", fontsize=13)
    plot_affinity(right.subplots(2, 2, sharex=True), affinity_rows, mode)
    figure.legend(handles=legend_handles(mode), loc="outside lower center", ncol=7 if mode == "bootstrap" else 4, frameon=False)
    figure.savefig(path, dpi=150)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11})
    rng = np.random.default_rng(0)
    pose_rows, affinity_rows = pose_values(rng), affinity_values(rng)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pose_rows).to_csv(args.out_dir / "results_at_a_glance_pose.csv", index=False)
    pd.DataFrame(affinity_rows).to_csv(args.out_dir / "results_at_a_glance_affinity.csv", index=False)
    make_figure(pose_rows, affinity_rows, "seeds", args.out_dir / "results_at_a_glance_seeds.png")
    make_figure(pose_rows, affinity_rows, "bootstrap", args.out_dir / "results_at_a_glance_bootstrap.png")


if __name__ == "__main__":
    main()

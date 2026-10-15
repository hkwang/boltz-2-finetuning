"""Results-at-a-glance figure for docs/report.md: pose (best-of-5 ligand RMSD) and affinity (Spearman, RMSE by split), with 95% bootstrap CIs over entries.

Run: python experiments/plot_results_glance.py [--out docs/figures/results_at_a_glance.png]
Reads experiments/results/mac1_eval/<run>/eval_per_entry.csv (pose; stock-head affinity on fine-tuned poses),
experiments/results/finetune_affinity/<run>/eval_step500.csv (tuned heads on fine-tuned poses) and
experiments/results/finetune_affinity/stockpose/<head>.csv (all heads on stock poses).
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
SEED_MARKERS = {2026: "o", 1: "s", 2: "^"}
POSE_RUNS = [
    ("stock", ["stock", "stock_seed1", "stock_seed2"]),
    ("full 3e-6, s1400", ["full_lr3e-6_step1400"]),
    ("full 1e-5, s1400", ["full_lr1e-5_step1400", "full_lr1e-5_step1400_seed1", "full_lr1e-5_step1400_seed2"]),
    ("full 1e-4, s1400", ["full_lr1e-4_step1400", "full_lr1e-4_step1400_seed1", "full_lr1e-4_step1400_seed2"]),
    ("full 3e-4, s1000", ["full_lr3e-4_step1000"]),
    ("full 3e-4, s2000", ["full_lr3e-4_step2000"]),
    ("LoRA 1e-4, s1400", ["lora_r16_lr1e-4_step1400", "lora_r16_lr1e-4_step1400_seed1", "lora_r16_lr1e-4_step1400_seed2"]),
    ("LoRA 3e-4, s2000", ["lora_r16_lr3e-4_step2000", "lora_r16_lr3e-4_step2000_seed1", "lora_r16_lr3e-4_step2000_seed2"]),
]
HEADS = ["stock", "aff_lr1e-5", "aff_lr2e-5", "aff_lr5e-5", "aff_nopair"]
POSE_COLORS = {"stock pose": "#4c72b0", "fine-tuned pose": "#dd8452"}


def seed_of(run: str) -> int:
    return int(run.rsplit("_seed", 1)[1]) if "_seed" in run else 2026


def median_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    draws = [np.median(values[rng.integers(0, len(values), len(values))]) for _ in range(BOOTSTRAPS)]
    return float(np.median(values)), *np.percentile(draws, [2.5, 97.5])


def affinity_ci(pred: np.ndarray, truth: np.ndarray, rng: np.random.Generator) -> dict[str, tuple[float, float, float]]:
    def spearman(p, y):
        return float(np.corrcoef(pd.Series(p).rank(), pd.Series(y).rank())[0, 1])

    def rmse(p, y):
        return float(np.sqrt(((p - y) ** 2).mean()))

    out = {}
    for name, stat in (("spearman", spearman), ("rmse", rmse)):
        draws = []
        for _ in range(BOOTSTRAPS):
            sample = rng.integers(0, len(pred), len(pred))
            draws.append(stat(pred[sample], truth[sample]))
        out[name] = (stat(pred, truth), *np.nanpercentile(draws, [2.5, 97.5]))
    return out


def by_pdb(table: pd.DataFrame, column: str) -> pd.Series:
    return table.set_index(table["record_id"].str.replace("mac1_eval_", ""))[column]


def affinity_predictions() -> dict[tuple[str, str], pd.Series]:
    predictions = {}
    for head in HEADS:
        predictions[("stock pose", head)] = by_pdb(pd.read_csv(AFFINITY / "stockpose" / f"{head}.csv"), "pred")
        if head == "stock":
            fine_tuned = pd.read_csv(EVAL / "full_lr1e-4_step1400" / "eval_per_entry.csv").set_index("pdb_id")["affinity_pred_value"]
        else:
            fine_tuned = by_pdb(pd.read_csv(AFFINITY / head / "eval_step500.csv"), "pred")
        predictions[("fine-tuned pose", head)] = fine_tuned
    return predictions


def plot_pose(axis: plt.Axes, rng: np.random.Generator) -> None:
    rows = []
    for label, runs in POSE_RUNS:
        for run in runs:
            values = pd.read_csv(EVAL / run / "eval_per_entry.csv")["rmsd_best"].dropna().to_numpy()
            rows.append((label, seed_of(run), *median_ci(values, rng)))
    labels = [label for label, _ in POSE_RUNS]
    offsets = {2026: -0.22, 1: 0.0, 2: 0.22}
    stock_mean = np.mean([row[2] for row in rows if row[0] == "stock"])
    axis.axvline(stock_mean, color="0.6", linestyle="--", linewidth=1.2, zorder=0, label="stock, mean of 3 seeds")
    for label, seed, median, low, high in rows:
        y = labels.index(label) + offsets[seed]
        color = "#4c72b0" if label == "stock" else "#55a868"
        axis.errorbar(median, y, xerr=[[median - low], [high - median]], fmt=SEED_MARKERS[seed], color=color, markersize=7, capsize=3, linewidth=1.5)
    for seed, marker in SEED_MARKERS.items():
        axis.plot([], [], marker, color="0.3", markersize=7, linestyle="none", label=f"seed {seed}")
    axis.set_yticks(range(len(labels)), labels)
    axis.invert_yaxis()
    axis.set_xlabel("Median best-of-5 ligand RMSD (Å)")
    axis.set_title("Pose: diffusion fine-tuning (45 entries)")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    axis.grid(axis="x", alpha=0.3)


def plot_affinity(axes: list[plt.Axes], rng: np.random.Generator) -> None:
    reference = pd.read_csv(EVAL / "stock" / "eval_per_entry.csv").set_index("pdb_id")
    predictions = affinity_predictions()
    width = 0.18
    for column, split in enumerate(("val", "test")):
        entries = reference.index[(reference["ic50_qualifier"] == "=") & (reference["split"] == split)]
        truth = reference.loc[entries, "log10_ic50_uM"].to_numpy()
        for row, metric in enumerate(("spearman", "rmse")):
            axis = axes[row][column]
            for pose_index, pose in enumerate(POSE_COLORS):
                for head_index, head in enumerate(HEADS):
                    value, low, high = affinity_ci(predictions[(pose, head)].loc[entries].to_numpy(), truth, rng)[metric]
                    x = head_index + (pose_index - 0.5) * 2 * width
                    axis.errorbar(x, value, yerr=[[value - low], [high - value]], fmt="o", color=POSE_COLORS[pose], markersize=7, capsize=3,
                                  linewidth=1.5, label=pose if head_index == 0 else None)
            axis.set_xticks(range(len(HEADS)), ["stock" if head == "stock" else head.replace("aff_", "") for head in HEADS])
            axis.set_xlim(-0.6, len(HEADS) - 0.4)
            axis.grid(axis="y", alpha=0.3)
            axis.set_title(f"{split} ({len(entries)} IC50s)")
            axis.set_ylabel("Spearman ρ" if metric == "spearman" else "RMSE (log10 µM)")
            if row == 1:
                axis.set_xlabel("Affinity head (step 500)")
    axes[0][0].legend(loc="lower right", frameon=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("docs/figures/results_at_a_glance.png"))
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "legend.fontsize": 11, "figure.titlesize": 14})
    rng = np.random.default_rng(0)
    figure = plt.figure(figsize=(15, 7.5), layout="constrained")
    left, right = figure.subfigures(1, 2, width_ratios=[1, 1.35])
    plot_pose(left.subplots(), rng)
    right.suptitle("Affinity: head fine-tuning, stock vs fine-tuned (full 1e-4, s1400) poses")
    plot_affinity(right.subplots(2, 2, sharex=True), rng)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Swarm waves 1 and 2 at a glance, in the style of docs/figures/results_at_a_glance_{seeds,bootstrap}.png.

Two figures with the same numbers and different error bars:
  swarm_summary_seeds.png      mean over prediction seeds (2026, 1, 2) with a 95% t-interval over seeds (n = 3); small dots are the seeds.
                               The paired panel is the per-seed median paired change (swarm_summary_paired_seeds.csv).
  swarm_summary_bootstrap.png  one marker per seed, each with a 95% bootstrap interval over held-out entries (2000 resamples); the
                               paired panel is the per-entry change averaged over seeds, with a bootstrap interval.

Each agent's final model (lowest validation best-of-5 on the board, plus agent1's base) was scored once on the 24 test entries after
the swarm ended (operator jobs PBS 7755524-7755535). Val (21, from the board runs) and test (24) are pooled to the same 45 entries as
docs/report.md. Left: median best-of-5 ligand RMSD per predict seed with a 95% bootstrap interval over entries (2000 resamples).
Middle: paired change against the best pre-agent fine-tune (lowest 3-seed mean of the report runs), per entry averaged over seeds,
median over entries, with a 95% bootstrap interval.
Right: affinity Spearman rho of `affinity_pred_value` against log10 IC50 on the 40 measured (`=`) IC50s, one marker per seed with a 95%
bootstrap interval. No agent trained the affinity head, so every row is the stock head on that model's poses; the dotted line is the
best tuned head of docs/figures/results_at_a_glance (aff_lr5e-5 on full 1e-4 poses, mean over seeds, read from results_at_a_glance_affinity.csv).

Run: python experiments/plot_swarm_summary.py docs/rescore/final_tables [--out-dir docs/figures]
TABLES_DIR holds eval_per_entry.csv copies: ref/<run>[_seed<S>].csv (report runs, 45 entries), val/<agent>__<label>_seed<S>.csv and
test/<agent>__<label>_seed<S>.csv (swarm models).
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BOOTSTRAPS = 2000
SEEDS = (2026, 1, 2)
SEED_MARKERS = {2026: "o", 1: "s", 2: "^"}
T_95 = {3: 4.303}
REPORT_RUNS = [
    ("stock", "stock"),
    ("full 3e-6, s1400", "full_lr3e-6_step1400"),
    ("full 1e-5, s1400", "full_lr1e-5_step1400"),
    ("full 1e-4, s1400", "full_lr1e-4_step1400"),
    ("full 3e-4, s1000", "full_lr3e-4_step1000"),
    ("full 3e-4, s2000", "full_lr3e-4_step2000"),
    ("LoRA 1e-4, s1400", "lora_r16_lr1e-4_step1400"),
    ("LoRA 3e-4, s2000", "lora_r16_lr3e-4_step2000"),
]
SWARM_RUNS = [
    ("a1 cosine+EMA base (w1)", "agent1__cosine_ema_s0", "#8172b3"),
    ("a1 cosine raw (w1)", "agent1__cosine_raw_s0", "#8172b3"),
    ("a3 contact loss (w1)", "agent3__contacts_ema1400_s0", "#8172b3"),
    ("a6 cond. gradients (w1)", "agent6__enabled_bridge_ema_s0", "#8172b3"),
    ("a8 symmetry graft (w1)", "agent8__symmetry_mean_trunkgraft", "#8172b3"),
    ("a2 high-res 196 (w2)", "agent2__resolution1A_ema1400_s0", "#c44e52"),
    ("a4 high-res 196, fixed (w2)", "agent4__b2_resolution1A_s0_aliasfix", "#c44e52"),
    ("a5 fresh 2800 steps (w2)", "agent5__b2_fresh2800_s0", "#c44e52"),
    ("a7 fresh 2800, fixed (w2)", "agent7__aliasfix_fresh2800_s0", "#c44e52"),
    ("a9 iid-noise control (w2)", "agent9__b2_iid1400_s0_aliasfix", "#c44e52"),
]


def tables(root: Path) -> dict[str, dict[int, pd.DataFrame]]:
    """{label: {seed: eval_per_entry rows of the 45 held-out entries, indexed by pdb_id}}."""
    out = {}
    for label, run in REPORT_RUNS:
        out[label] = {seed: pd.read_csv(root / "ref" / f"{run}{'' if seed == 2026 else f'_seed{seed}'}.csv").set_index("pdb_id") for seed in SEEDS}
    for label, run, _ in SWARM_RUNS:
        out[label] = {}
        for seed in SEEDS:
            val = pd.read_csv(root / "val" / f"{run}_seed{seed}.csv")
            test = pd.read_csv(root / "test" / f"{run}_seed{seed}.csv")
            out[label][seed] = pd.concat([val[val["split"] == "val"], test[test["split"] == "test"]]).set_index("pdb_id")
    return out


def spearman(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.corrcoef(pd.Series(pred).rank(), pd.Series(truth).rank())[0, 1])


def spearman_interval(pred: np.ndarray, truth: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    draws = []
    for _ in range(BOOTSTRAPS):
        sample = rng.integers(0, len(pred), len(pred))
        draws.append(spearman(pred[sample], truth[sample]))
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return spearman(pred, truth), float(low), float(high)


def median_interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    draws = [np.median(values[rng.integers(0, len(values), len(values))]) for _ in range(BOOTSTRAPS)]
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(np.median(values)), float(low), float(high)


def seed_interval(values: list[float]) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    half = T_95[len(values)] * float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return mean, mean - half, mean + half


def draw(axis: plt.Axes, position: float, rows: list[dict], key: str, mode: str, color: str) -> None:
    """Seeds mode: one diamond (mean, t-interval over seeds) plus small seed dots. Bootstrap mode: one marker and bar per seed."""
    if mode == "seeds":
        mean, low, high = seed_interval([row[key] for row in rows])
        axis.errorbar(mean, position, xerr=[[mean - low], [high - mean]], fmt="D", color=color, markersize=6, capsize=3, linewidth=1.8)
        for row in rows:
            axis.plot(row[key], position, SEED_MARKERS[row["seed"]], color=color, markersize=3.5, alpha=0.55, markeredgewidth=0)
    else:
        for row in rows:
            offset = {2026: -0.22, 1: 0.0, 2: 0.22}[row["seed"]]
            axis.errorbar(row[key], position + offset, xerr=[[row[key] - row["low"]], [row["high"] - row[key]]], fmt=SEED_MARKERS[row["seed"]],
                          color=color, markersize=4.5, capsize=2, linewidth=1.1)


def make_figure(rows: list[dict], deltas: list[dict], seed_deltas: list[dict], affinity: list[dict], context: dict, mode: str, path: Path) -> None:
    labels, colors = context["labels"], context["colors"]
    figure = plt.figure(figsize=(13.33, 7.5), layout="constrained")
    if mode == "seeds":
        figure.suptitle("Swarm waves 1-2 vs simple fine-tuning: 95% CI over prediction seeds (n = 3), not entries", fontsize=13, color="0.25")
    else:
        figure.suptitle("Swarm waves 1-2 vs simple fine-tuning: 95% bootstrap CI over the 45 held-out entries", fontsize=13, color="0.25")
    left, right, third = figure.subplots(1, 3, sharey=True, width_ratios=[1.25, 0.9, 0.9])
    stock = np.mean([row["value"] for row in rows if row["label"] == "stock"])
    left.axvline(stock, color="0.6", linestyle="--", linewidth=1.2, zorder=0)
    for index, label in enumerate(labels):
        draw(left, index, [row for row in rows if row["label"] == label], "value", mode, colors[label])
    left.set_yticks(range(len(labels)), labels)
    left.invert_yaxis()
    left.set_xlabel("Best-of-5 ligand RMSD, median (Å)")
    left.set_title("Pose (n = 45, val + test)")
    left.grid(axis="x", alpha=0.3)
    right.axvline(0, color="#55a868", linestyle="--", linewidth=1.2, zorder=0)
    if mode == "seeds":
        for index, label in enumerate(labels):
            draw(right, index, [row for row in seed_deltas if row["label"] == label], "delta", mode, colors[label])
        right.set_title("Paired per entry, per seed")
    else:
        for index, row in enumerate(deltas):
            right.errorbar(row["delta"], index, xerr=[[row["delta"] - row["low"]], [row["high"] - row["delta"]]], fmt="D", color=colors[row["label"]],
                           markersize=5, capsize=3, linewidth=1.5)
        right.set_title("Paired per entry, mean of 3 seeds")
    right.set_xlabel(f"Paired change vs {context['reference']} (Å)")
    right.grid(axis="x", alpha=0.3)
    stock_rho = np.mean([row["spearman"] for row in affinity if row["label"] == "stock"])
    third.axvline(stock_rho, color="0.6", linestyle="--", linewidth=1.2, zorder=0)
    third.axvline(context["tuned"], color="#dd8452", linestyle=":", linewidth=1.6, zorder=0)
    for index, label in enumerate(labels):
        draw(third, index, [row for row in affinity if row["label"] == label], "spearman", mode, colors[label])
    third.set_xlabel("Affinity Spearman ρ (stock head)")
    third.set_title(f"Affinity (n = {context['measured']} IC50s)")
    third.grid(axis="x", alpha=0.3)
    for axis in (left, right, third):
        axis.axhline(len(REPORT_RUNS) - 0.5, color="0.75", linewidth=0.8)
    handles = [plt.Line2D([], [], color=color, marker="s", linestyle="none", markersize=8, label=name)
               for name, color in (("stock", "#4c72b0"), ("simple FT (report)", "#55a868"), ("swarm wave 1", "#8172b3"), ("swarm wave 2", "#c44e52"))]
    handles.append(plt.Line2D([], [], color="0.6", linestyle="--", label="stock"))
    handles.append(plt.Line2D([], [], color="#dd8452", linestyle=":", linewidth=1.6, label="best tuned affinity head"))
    if mode == "seeds":
        handles.append(plt.Line2D([], [], color="0.3", marker="D", linestyle="-", markersize=6, label="seed mean ± 95% CI"))
    size = 4 if mode == "seeds" else 6
    handles += [plt.Line2D([], [], color="0.3", marker=marker, linestyle="none", markersize=size, label=f"seed {seed}") for seed, marker in SEED_MARKERS.items()]
    figure.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, handlelength=1.2, columnspacing=0.8, handletextpad=0.3,
                  fontsize=10)
    figure.savefig(path)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tables", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 10.5,
                         "legend.fontsize": 11, "savefig.dpi": 200})
    rng = np.random.default_rng(0)
    frames = tables(args.tables)
    data = {label: {seed: frame["rmsd_best"] for seed, frame in runs.items()} for label, runs in frames.items()}
    ids = data["stock"][2026].index
    assert len(ids) == 45 and all(len(series.reindex(ids).dropna()) == 45 for runs in data.values() for series in runs.values())
    reference_table = frames["stock"][2026]
    measured = reference_table.index[reference_table["ic50_qualifier"] == "="]
    truth = reference_table.loc[measured, "log10_ic50_uM"].to_numpy()
    colors = {label: "#4c72b0" if label == "stock" else "#55a868" for label, _ in REPORT_RUNS} | {label: color for label, _, color in SWARM_RUNS}
    labels = [label for label, _ in REPORT_RUNS] + [label for label, _, _ in SWARM_RUNS]
    means = {label: np.mean([data[label][seed].reindex(ids).median() for seed in SEEDS]) for label, _ in REPORT_RUNS if label != "stock"}
    reference = min(means, key=means.get)
    rows, deltas, seed_deltas, affinity = [], [], [], []
    for label in labels:
        for seed in SEEDS:
            value, low, high = median_interval(data[label][seed].reindex(ids).to_numpy(), rng)
            rows.append({"label": label, "seed": seed, "value": value, "low": low, "high": high})
        change = np.mean([data[label][seed].reindex(ids).to_numpy() - data[reference][seed].reindex(ids).to_numpy() for seed in SEEDS], axis=0)
        value, low, high = median_interval(change, rng)
        deltas.append({"label": label, "delta": value, "low": low, "high": high})
        for seed in SEEDS:
            value, low, high = spearman_interval(frames[label][seed].loc[measured, "affinity_pred_value"].to_numpy(), truth, rng)
            affinity.append({"label": label, "seed": seed, "n": len(measured), "spearman": value, "low": low, "high": high})
    pd.DataFrame(rows).to_csv(args.out_dir / "swarm_summary_pose.csv", index=False)
    pd.DataFrame(deltas).to_csv(args.out_dir / "swarm_summary_paired.csv", index=False)
    pd.DataFrame(affinity).to_csv(args.out_dir / "swarm_summary_affinity.csv", index=False)
    glance = pd.read_csv(args.out_dir / "results_at_a_glance_affinity.csv")
    tuned = glance[(glance["metric"] == "spearman") & (glance["pose"] == "fine-tuned pose") & (glance["head"] == "aff_lr5e-5")]["value"].mean()

    for seed in SEEDS:
        for label in labels:
            change = data[label][seed].reindex(ids).to_numpy() - data[reference][seed].reindex(ids).to_numpy()
            seed_deltas.append({"label": label, "seed": seed, "delta": float(np.median(change))})
    pd.DataFrame(seed_deltas).to_csv(args.out_dir / "swarm_summary_paired_seeds.csv", index=False)
    context = {"labels": labels, "colors": colors, "reference": reference, "tuned": tuned, "measured": len(measured)}
    make_figure(rows, deltas, seed_deltas, affinity, context, "seeds", args.out_dir / "swarm_summary_seeds.png")
    make_figure(rows, deltas, seed_deltas, affinity, context, "bootstrap", args.out_dir / "swarm_summary_bootstrap.png")

if __name__ == "__main__":
    main()

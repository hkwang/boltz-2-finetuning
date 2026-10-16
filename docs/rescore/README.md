# Batch-1 rescore: all five samples

Written 2026-10-16 (operator) before swarm batch 2. Every stored validation prediction of batch 1 (350 prediction directories: 95 board candidates plus stock, each at predict seeds 2026/1/2) was rescored per diffusion sample with the frozen scoring functions (`experiments/rescore_samples.py`, `experiments/run_rescore_samples.sh`, PBS 7753493; summary `experiments/summarize_rescore.py`, PBS 7753527). The rescore reproduces `rmsd_top1` and `rmsd_best` of all 321 checked `eval_per_entry.csv` tables exactly (max |diff| 0).

## Files

| File | Content |
|---|---|
| `batch1_per_entry.csv` | one row per (agent, prediction dir, val entry): `top1`, `best`, `median`, `mean`, `worst` ligand RMSD over the 5 samples, `frac_samples_below_2A` |
| `batch1_per_candidate.csv` | one row per board candidate: each metric as the median over the 21 entries of the 3-seed mean (fraction metric: mean), paired difference vs stock `d_<metric>` with a 2000-resample bootstrap over entries (`lo_`, `hi_`), and `seed_sd_median` |
| `batch1_matched_contrasts.txt` | add-on vs matched base contrasts, and the per-entry change of the cosine+EMA base vs stock |

## Metrics

- **`mean`** (mean ligand RMSD over the 5 samples): the headline for batch 2. It uses every sample, so it measures the sampler that training changes, it is not affected by ranking, and it is less noisy than a single pose.
- `best` (best-of-5) rewards one lucky sample; `top1` depends on the confidence ranking; `frac_samples_below_2A` is the success rate per sample.

Stock (val, 3-seed mean): `top1` 2.43, `best` 1.49, `median` 2.29, `mean` 2.26 A, 49 % of samples below 2 A.

## Findings

1. **Training improves the sampled poses; ranking does not.** Most fine-tuned models cut the per-entry mean RMSD over the 5 samples by 0.12-0.22 A (`r90` -0.215 [-0.271, -0.019], `r47` -0.194, `r49` -0.179, `r65` -0.175) and raise the fraction of samples below 2 A from 49 % to about 61-66 % (`r47` +0.171 [+0.070, +0.292]). Top-1 moves much less (-0.02 to -0.09 A).
2. **Almost all of that gain is one recipe: full diffusion FT at about 3e-5 for about 1400 steps.** Against the matched cosine+EMA base (two training seeds), every add-on is within +/-0.02 A on mean RMSD: symmetry +0.002 [-0.001, +0.005], conditioning gradients -0.008 [-0.086, -0.001], pocket-frame ligand loss w4 +0.022 [+0.001, +0.047] (slightly worse), contact loss (seed 0 only) +0.022 [-0.019, +0.081], ligand weight 50 (seed 0 only) +0.004 [-0.023, +0.018]. EMA itself makes no difference against the raw final weights (+0.001 [-0.001, +0.003]). The leaderboard differences between these runs are mostly training-seed and selection noise.
3. **Short or narrow fine-tunes give about a third of the gain.** 300-500-step runs at 1e-5 (high resolution, uniform controls, scaffold balancing, noise schedules, L2-SP) reach -0.03 to -0.09 A. Training only the conditioning, trunk, atom encoder/decoder or confidence head gives about 0 (`r7`, `r9`, `r16`, `r17`, `r43`).
4. **The gain is concentrated in a few entries.** For the cosine+EMA base vs stock, 6 of 21 entries improve by more than 0.25 A (7HCN -2.09, 7HF9 -1.22, 7HCX -0.96, 7HF6 -0.62, 7HF1 -0.60, 13RS -0.28), 14 change by less than 0.25 A and one gets worse (7HCG +0.61). Entries such as 14AB (4.8 A), 7HCG (5.5-6.1 A) and 7HFD (3.7 A) are wrong in every sample for every model: a binding-mode problem that no batch-1 recipe touches.
5. **Data curation by resolution has a small effect at matched steps.** High resolution (<= 1 A, 196 crystals) vs all 557, both 300 steps at 1e-5: -0.016 [-0.079, -0.010] mean RMSD (2 vs 1 training seeds).
6. **Training seed matters about as much as most add-ons.** Cosine+EMA seed 1 vs seed 0: -0.001 [-0.029, +0.011] mean, -0.009 best; leaderboard medians of single runs move 0.1-0.3 A with predict seed (`seed_sd_median`).

## Implications for batch 2

- Use `mean` RMSD over the 5 samples (and `frac_samples_below_2A`) as the headline; compare each change with a **matched** base (same steps, LR, schedule, training seeds), not with stock.
- The recipe's add-ons are exhausted at this scale. The open questions are what limits the cosine+EMA base (data, steps, scale), and the always-wrong entries (14AB, 7HCG, 7HFD, 7HCJ, 7HFE): their ligands, occupancy and nearest training chemotypes are a cheap diagnostic.

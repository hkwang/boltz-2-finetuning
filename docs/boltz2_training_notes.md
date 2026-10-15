# Boltz-2 training: what the code supports and how to format our data

Read on 2026-10-14. Nothing here was run. Sources: installed `boltz==2.2.1`, upstream `jwohlwend/boltz` (HEAD b1ebfc46, release v2.2.1), and three public repos
listed at the bottom. Public-repo code was read, not executed or audited.

## 1. What installed Boltz 2.2.1 gives us

- `Boltz2.training_step` (`boltz/model/models/boltz2.py:793`) sums four losses: confidence, diffusion, distogram and (optional) B-factor, weighted by
  `training_args.{confidence,diffusion,distogram,bfactor}_loss_weight`. Structure losses run only if `structure_prediction_training` is true;
  the confidence loss only if `confidence_prediction` is true.
- **It contains no affinity loss.** `forward` runs the `AffinityModule` (lines ~608-700), but nothing in `training_step` supervises it, and
  `boltz/model/loss/` has no affinity file. So the affinity head cannot be fine-tuned with stock Boltz 2.2.1.
- `boltz/data/module/trainingv2.py` is not a Boltz-2 data pipeline. It imports the v1 `BoltzFeaturizer` and v1 `Tokenizer`, takes no `mol_dir`, and
  never passes `molecules`/`random` to the featurizer, which `Boltz2Featurizer.process` (`featurizerv2.py:2160`) requires. Treat it as unusable for Boltz-2 as shipped.
- Upstream ships only Boltz-1 training: `scripts/train/train.py` plus `configs/{structure,confidence,full}.yaml`. `docs/training.md` says Boltz-2 training info is "coming soon".
- What a training sample is (from `trainingv2.py` and `data/types.py`): a directory with `manifest.json` (a list of `Record`), `structures/{id}.npz`
  (a `Structure`/`StructureV2` with atoms, bonds, residues, chains, interfaces, mask), and `msa/{msa_id}.npz`. A `Record` has `id`, `structure`
  (`StructureInfo`), `chains` (`ChainInfo`, with `msa_id`), `interfaces`, and an optional `affinity` (`AffinityInfo(chain_id, mw)`).
  Validation is by ID: `DatasetConfig.split` is a text file of record IDs; those go to validation, the rest to training.
- `boltz predict` already contains the converter from YAML to this format: `process_inputs` in `boltz/main.py` writes `processed/{manifest.json, records/, structures/, msa/}`.
  We already have such output under `experiments/results/mac1_cofold/boltz_results_inputs/processed/`. For affinity-only work, reuse this path instead of writing our own parser.

## 2. The only public Boltz-2 affinity fine-tuning recipe

`molecularinformatics/Boltz2_affinity` (a fork of Boltz; `docs/affinity_finetuning.md`) trains **only the affinity heads**. `ohuelab/BoltzFT` is the paper's pipeline on top of it,
and `sermare/molasses` is a third-party reproduction. Pipeline:

1. One YAML per (protein, ligand) with the label under `properties: - affinity:`. Fork fields: `binder`, `value` (continuous), `is_binder` (0/1; the molasses patch also allows a soft value in [0, 1]), `assay_id`, `measurement_type`, `confidence` (sample weight).
2. `boltz predict --write_embeddings` (recycling 5, 200 sampling steps, 5 diffusion samples in the doc) co-folds each complex and caches trunk `z`, `s_inputs` and the best-ipTM coordinates.
3. `scripts/process/data_collector.py` consolidates predictions and YAML labels into `consolidated/{structures,records,msa,mols,embeddings,manifest.json}`.
4. `scripts/train/trainv2.py scripts/train/config_aff/config.yaml` (Hydra) loads `boltz2_aff.ckpt` and trains with `boltz.data.module.trainingv3` (a Boltz-2 data module that adds `mol_dir`, `extra_mols_dir`, `embeddings_dir`).
   Config: `affinity_only_finetuning: true`, `structure_prediction_training: false`, trunk/diffusion/confidence frozen, `affinity_ensemble: true`, bf16-mixed, EMA 0.999, grad clip 10, `accumulate_grad_batches: 16`, batch size 1.
5. Loss (`src/boltz/model/loss/affinity.py`): Huber on continuous values (`affinity_absolute_weight` 0.5), pairwise-difference Huber within an `assay_id` (weight 2.0, up to 50 pairs per assay), focal loss on binder/decoy (weight 1.0, alpha 0.25, gamma 2). A term is skipped when its label mask is empty.
6. Validation is an explicit ID file (`paths.split_file`), one complex name per line.

Reproduction hyperparameters in `molasses/pipeline_local/train_headft_seed.sh` for binary-only head fine-tuning: absolute and pairwise weights 0, binary weight 1, `max_lr` 2e-5, 5 epochs, accumulate 4, 40-300 training compounds.
The fork's default is `max_lr` 1e-4 with 200 warmup steps.

## 3. What this means for our data

- **The experimental structure is not used for affinity-only training.** The pose is re-predicted by Boltz and cached. So `structure_loss_weight`, the deposited ligand pose, and the apo 7KQO pairing for non-hits do not enter this recipe.
  Non-hits become ordinary (protein, SMILES) YAMLs with `is_binder: 0`.
- **Row-to-label mapping for the affinity recipe** (our tables, `docs/data_schema.md`):

  | Table (train rows) | Rows | Label |
  |---|---|---|
  | `mac1_splits.csv` | 125 | `value` from `ic50_uM`, but check units/transform below; censored (`>`) rows have no exact value |
  | `mac1_fragment_hit_records.csv` | 183 with `affinity_binary` = 1 | `is_binder: 1` |
  | `mac1_nonhit_records.csv` | 2347 | `is_binder: 0` |

  All of it is one protein with one `assay_id` per source, so pairwise loss only compares within a source.
- **Open questions that need the fork's code before we encode values:** the units and transform expected for `value` (Boltz-2 outputs log10 IC50 in uM; the fork doc's example is just `0.5`), and how censored values should be handled (probably drop from the continuous loss, keep for binary).
  I did not read `trainingv3.py` or the fork's featurizer changes closely enough to settle this.
- **Structure (diffusion) fine-tuning on our 557 structures has no public recipe.** It would need a Boltz-2 data module with `mol_dir` and `Boltz2Featurizer` (the fork's `trainingv3.py` is the closest thing), structures processed to `StructureV2`,
  and `structure_prediction_training: true` with the diffusion/distogram weights set in `training_args`. Boltz-2's training_step supports that part; the missing piece is the data pipeline and a training entry point.
  Decide whether we want this at all; the binding-affinity task only needs the affinity head.
- **Class balance:** 2347 negatives vs 183 positives and 125 IC50 pairs. The fork's focal loss (alpha 0.25) and the `confidence` field (sample weight) are the available knobs; subsampling is another. The reproduction used 40-300 compounds per target, so our negative count is far larger.
- **Leakage:** the Boltz-2 cutoff and ligand-cluster rules in `docs/datasets.md` still apply. Embeddings are cached per complex, so build them only for the chosen splits.

## 4. Suggested next steps

1. Pin the fork (`molecularinformatics/Boltz2_affinity` @ `bc06a0b`, the commit the reproduction used) in a separate environment, since it replaces the installed `boltz`. Do not edit `~/venvs/boltz2`.
2. Read the fork's `trainingv3.py`/featurizer for the `value` units and the censored-value handling.
3. Write `experiments/build_mac1_affinity_yamls.py`: tables to YAMLs with `properties: affinity`, plus train and val ID files.
4. Run `boltz predict --write_embeddings` as a PBS job (Polaris compute nodes have no network, so stage the fork and checkpoints first), then the consolidation and training scripts.
5. Zero-shot baseline first: the existing co-folding run on the 25 test entries (the config currently lists only 6).

## Sources

- Upstream: https://github.com/jwohlwend/boltz (`docs/training.md`, `scripts/train/`)
- Affinity fork: https://github.com/molecularinformatics/Boltz2_affinity (`docs/affinity_finetuning.md`, `scripts/train/config_aff/`, `src/boltz/model/loss/affinity.py`, `src/boltz/data/module/trainingv3.py`)
- Paper pipeline: https://github.com/ohuelab/BoltzFT (`pipeline/`, `patches/boltz2_affinity.patch`)
- Reproduction: https://github.com/sermare/molasses (`pipeline_local/train_headft_seed.sh`, `docs/PIPELINE_NOTES.md`)
- Local copies fetched to `/tmp/ext` (not persistent).

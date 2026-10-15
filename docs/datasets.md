# Dataset log

Shared running notes on candidate structure-affinity datasets. Append new entries; record
source, date retrieved, counts, units, and known leakage/quality issues.

## Mac1 (SARS-CoV-2 NSP3 macrodomain), Fraser/Correy labs — retrieved 2026-10-14

Rebuild: `~/venvs/boltz2/bin/python experiments/build_mac1_dataset.py`
(downloads missing supplements into `data/raw/mac1/`, writes `data/processed/mac1_affinity.csv`).

**Label:** HTRF peptide-displacement IC50 in µM (`ic50_uM`), with `ic50_qualifier` (`=` or `>`; `>` means
censored, not a measured value) and `ic50_raw` as published. Ki (Cheng-Prusoff from IC50) is kept only where
the source gives it. Cells with two values (stereoisomer pairs, PNAS) have `ic50_uM` empty and raw text kept.
One table row = one (PDB entry, ligand) pair; `pdb_in_rcsb` records whether the ID resolves in RCSB.

| Source (`source`) | Paper / file | Rows | With IC50 | Public in RCSB | Usable (public + IC50) |
|---|---|---|---|---|---|
| `elife110475` | Kim, Correy ... Fraser, eLife RP110475 (doi 10.7554/eLife.110475), Supp. file 3 | 491 | 179 | 26 | 22 |
| `scienceadv` | Correy et al., Sci. Adv. 2025, doi 10.1126/sciadv.ads7187, Table S2 (sheet B_compounds) | 140 | 65 | 140 | 65 |
| `pnas` | Gahbauer et al., PNAS 2023, doi 10.1073/pnas.2212931120, Dataset S1 | 153 | 70 | 153 | 70 |
| `schuller2021` | Schuller et al., Sci. Adv. 2021, doi 10.1126/sciadv.abf8711, Data file S1 sheet 12 | 8 | 8 | 8 | 8 |
| `jmedchem` | Jaishankar et al., J. Med. Chem. 2026 (AVI-6451), doi 10.1021/acs.jmedchem.5c02933, Tables 1-5 (PMC12788873) | 17 | 17 | 17 | 17 |

Usable rows: 182 over 171 unique PDB IDs after adding `schuller2021` (was 174 over 163). IC50 range 0.02–1000 µM; 7 are `>` censored.
Raw supplements and the J. Med. Chem. PMC XML: `data/raw/mac1/`.

**How `jmedchem` and the PDB gap-fill work.** The J. Med. Chem. tables give IC50 by AVI compound ID but no PDB IDs
(the paper's IDs are `13RI`–`13RY`, `9YKH`). The script queries RCSB for Mac1 entries titled "...in complex with AVI-nnnn",
maps AVI ID → PDB ID, and takes the SMILES from the ligand actually deposited in that entry (RCSB `SMILES_stereo`),
so these 17 rows have SMILES that match the modelled ligand. The same AVI→PDB map also fills PDB IDs that are blank in
eLife Supp. 3 / Sci. Adv. S2 (1 extra eLife row). Rows mapped this way pair IC50 and structure by AVI ID, not by a published PDB column.

**eLife 103484 (Mac1 target paper)** adds no rows: its IC50s appear only in figures and in prose (e.g. AVI-1500 ~120 nM, AVI-4206 ~20 nM),
and Supp. file 1 lists X-ray statistics only. Its 10 PDB IDs (9CXY, 9CXZ, 9CY0, 7HC4–7HCA) are already
covered by the other sources, with IC50 from the J. Med. Chem. tables where reported.

### Splits — `data/processed/mac1_splits.csv`

Rebuild: `PYTHONPATH=src ~/venvs/boltz2/bin/python experiments/make_mac1_splits.py` (seed 0, defaults below);
invariants are checked by `PYTHONPATH=src ~/venvs/boltz2/bin/python testing/test_splits.py`. Code: `src/boltz2_finetuning/splits.py`.

1. **Dedup to one row per PDB ID** (163 pairs). On conflict a measured (`=`) IC50 beats a censored (`>`) one, then source
   priority `jmedchem > scienceadv > pnas > elife110475`. `ic50_min_uM`/`ic50_max_uM`/`sources` keep the cross-source spread
   (3 PDB IDs disagree: 7HF7, 9D6B, 7HC6). In 7 PDB entries the sources give different SMILES strings; 6 are the same molecule written differently (identical InChIKey),
   and 7HF7 differs in stereochemistry between eLife and Sci. Adv. The highest-priority source's SMILES is kept.
2. **Leakage rule.** `release_date` (RCSB) on or before the Boltz-2 cutoff `2023-06-01` marks a structure as possibly seen in
   pretraining (`pre_cutoff`; 71 pairs; the paper's Supp. 6 list is a subset of these). Pre-cutoff pairs are train only.
   The cutoff has no ambiguity in this data (no releases between 2023-05 and 2023-08).
3. **Ligand clusters.** Butina clustering on Morgan r=2 / 2048-bit Tanimoto, similarity ≥ 0.5 (41 clusters on 163 pairs). A cluster is never
   split, and any cluster containing a pre-cutoff pair is train only. So val/test hold neither pretraining structures nor close analogues of them.
4. **Assignment.** Remaining clusters are shuffled with the seed and filled into test then val (15% each by count).

Resulting split (seed 0): train 115 (71 pre-cutoff, 2 censored), val 24 (4 censored, median IC50 53.5 µM), test 24 (1 censored, median 3.35 µM).

**Limitations.** val and test are small (24 pairs, 6 clusters each), so metrics will be noisy — report seed-to-seed spread across
several seeds. Their potency distributions differ (see medians). Everything in val/test is post-2025 lab-internal series
(mostly Sci. Adv., J. Med. Chem., eLife), so it measures generalisation to later analogues of the same target, not to new proteins.
`>` rows should be excluded from or handled specially in regression metrics. No protein/pocket diversity: all rows are Mac1.
When the `15xx` eLife structures release, re-run both scripts: splits will change (more held-out data), so pin the table
version used in any reported result.

### Structures for diffusion-head training — `data/structures/mac1/*.cif`, `data/processed/mac1_structures.csv`

Plan: train the diffusion (structure) head on all Mac1 structures, and the affinity head only on the IC50 pairs.
Rebuild: `~/venvs/boltz2/bin/python experiments/fetch_mac1_structures.py` (login node; mmCIFs are skipped if already present).

- The script scans every cell of `data/raw/mac1/*.xlsx` (and the J. Med. Chem. XML) for 4-character PDB IDs: 1153 candidates, 687 public in RCSB.
- It keeps entries with a ≥100-residue exact match to the Mac1 sequence of 5SQW: **603 structures** (60 MB, all X-ray, 0.80–2.50 Å, median 1.03 Å).
  The other 84 are not Mac1 and were dropped: AmpC, σ2, D4, Mpro, PLpro and CHIKV macrodomain IDs that Supp. 6 and the PNAS sheets also list.
- 163 have an IC50 pair (train 115 / val 24 / test 24); 440 have none. 411 of those 440 are PanDDA group depositions (ligand-bound or ground-state models from fragment screens).
- `diffusion_train` is False only for the 48 val/test structures, so held-out evaluation stays held out. 421 structures predate the 2023-06-01 Boltz-2 cutoff.
- Not available: the 465 unreleased eLife `15xx` structures. Re-run after they release.

**Caveats for the structure set.** Only the val/test PDB entries are excluded; the 440 unpaired structures were not clustered against the
val/test ligands, so close analogues of held-out ligands may be in diffusion training. Crystal forms are near-identical across entries, so
structure-level held-out pose accuracy is optimistic by construction. Multi-ligand entries (about 210 have ≥2 non-polymer entities, e.g. ions, glycerol, ADPr)
need the intended ligand chosen when building inputs. `files.rcsb.org` needs `certifi` (the cluster CA bundle lacks its intermediate certificate).

### J. Med. Chem. 2026 supporting information (supplied manually) — `data/raw/mac1/jmedchem_2026_si_00{1,2}`

- `si_001.csv`: 61 compounds with SMILES, Mac1 HTRF IC50 (µM) and ADME columns (latin-1, CRLF). `si_002.pdf`: dose-response curves
  (Fig. S1), PK tables, and X-ray statistics (Table S3) for 13RI–13RY and 9YKH.
- **No new usable pairs.** 24 of the 61 compounds have a public structure, and all 24 were already in the table. The other 37 have no released
  structure (8 of them match unreleased eLife `15xx` entries, e.g. AVI-4684 ↔ 15FD) and would only add pairs once released.
- IC50 values in the CSV match the table except two: AVI-4206 (CSV 0.0061 µM, table 0.020) and AVI-6451 (CSV 0.031, table 0.028).
  The article text gives ~20 nM for AVI-4206 and the table values come from the paper's Tables 1–5; this is unresolved, so the table still uses the paper tables.
  CSV also has censored/odd entries: AVI-6373 `< 0.012` (assay floor) and AVI-6253 `2.70E+04` µM. The CSV is not ingested by the build script yet.

### Caveats (read before training)

- **eLife structures are mostly unreleased.** The article says 468 new structures are deposited, but 465 of the 490 IDs in
  Supp. file 3 are `15xx` and return 404 from RCSB on 2026-10-14. Only 25 resolve (`13xx`, `14xx`, `7Hxx`, `7Ixx`, `9Dxx`).
  Re-run the build script later; `pdb_in_rcsb` and the usable count update automatically. 179 eLife rows have IC50, so up to
  ~157 more pairs may appear once released. Without coordinates, these rows cannot be used for structure labels.
- **Leakage.** The eLife paper is a Boltz-2 benchmark on structures deposited after training cutoffs (Boltz-2: 2023-06-30).
  Fine-tuning on these and then evaluating on the paper's benchmark would leak. Supp. file 6 lists PDB IDs per cutoff
  set (`data/raw/mac1/elife110475_supp6.xlsx`, 453 Mac1 IDs; 264 AF3/Chai-1, 189 Boltz-2 additional). Of the 174 usable rows, 69 are in this list
  (66 PNAS, 3 Sci. Adv.; 67 unique PDB IDs) and 105 are not. Treat these as likely already seen by Boltz-2 pretraining. Define the split before use.
- **Overlap between sources.** 10 PDB IDs occur in more than one usable row (5SQJ, 5SRY, 7HF7, 9D6B, 9D6H, 13RK, 13RL, 13RQ, 13RR, 7HC6).
  Most agree exactly. Differences: 7HF7 (70 vs 56 µM), 9D6B (4.2 vs 5.1 µM) and 7HC6 (0.55 vs 0.44 µM; the same compound under
  two AVI IDs, AVI-694 and AVI-3764). Deduplicate by PDB ID before splitting, and decide which source wins on conflict. PNAS has 2 PDB IDs with two ligands each.
- **Duplicates / non-unique mapping.** Multi-site PNAS entries like `5SOX (N3)` are truncated to the PDB ID; no IC50 for them.
- **Bias.** Potency distribution skews weak (fragments, early hits; median ~2–50 µM depending on source); potent late-stage
  compounds were published elsewhere (J. Med. Chem. 2026, AVI-6451, doi 10.1021/acs.jmedchem.5c02933; eLife 2025, doi 10.7554/eLife.103484).
- **Assay consistency.** All values are HTRF IC50 from the same lab, but assay conditions may vary across papers. Not yet verified
  across sources; not converted to pIC50 or Ki.
- **Ligand SMILES** are vendor/design SMILES, not extracted from the PDB ligand. Stereochemistry (racemates, isomer pairs
  in crystals) may not match the modelled ligand. Verify against the PDB before using.

### Not used / to follow up

- eLife Supp. file 3 was ingested only for PDB ID, SMILES and IC50; its co-folding/docking scores and RMSDs were not. Supp. files 1 (X-ray
  statistics, source paper per structure), 2 (32 MB similarity matrix) and 4 (H-bond tables) were inspected but not ingested.
  Supp. file 5 (AmpC/σ2/D4 docking hits) is not Mac1.
- Code for the eLife benchmark: https://github.com/jongbin99/cofold-em (input prep and analysis scripts).
- Still not scraped: Sci. Adv. 2021 fragment screen (doi 10.1126/sciadv.abf8711; mostly fragments, few IC50s), the
  bioRxiv preprint of the J. Med. Chem. paper (blocked by Cloudflare; the PMC text was sufficient), and the PanDDA
  depositions on Zenodo (10.5281/zenodo.7231822, 10.5281/zenodo.6974665), which are structures without IC50s. The J. Med. Chem.
  supplement was not open access via Europe PMC; it was later supplied manually (see below).
- No ChEMBL records were found for the three papers above.

## Mac1 non-hits — `data/processed/mac1_nonhits.csv` (retrieved 2026-10-14)

Rebuild: `~/venvs/boltz2/bin/python experiments/build_mac1_nonhits.py` (local file only, no network).
Source: Schuller et al., Sci. Adv. 2021 (doi 10.1126/sciadv.abf8711), Data file S1, `data/raw/mac1/abf8711_data_file_s1.xlsx`.
Not Boltz-affinity training labels: no PDB entry and no IC50. Use as negatives for binder-probability evaluation.

| `evidence` | Rows | Meaning |
|---|---|---|
| `xtal_nonhit` | 2316 | Fragment soaked into P43 and/or C2 crystals with no ligand density (sheets 8, 10), deduplicated by molecule. Weak negative: soak concentration (0.1–1 M stocks), packing and loop conformation limit sensitivity. 222 rows were not PanDDA-analysed (`pandda_analyzed=False`). |
| `docked_unconfirmed_htrf` | 39 | Docked compounds not confirmed by soaking, with DSF dTm and HTRF `pct_competition` at `htrf_top_conc_mM` (sheet 12). Stronger evidence than a crystallographic non-hit, but not thresholded here. |

Caveats: a molecule that is a hit in any crystal form is excluded. The libraries screened are fragments (small), so their distribution differs from the lead-like AVI series in `mac1_affinity.csv`.
Some Enamine isomers were mis-assigned (supplementary text; ZINC901391520, ZINC82473428, ZINC89254160), so their SMILES may be wrong.
The sheet marks 6 rows as having no dataset ID (`p43_no_id_*`); they have SMILES but no resolution and are kept as non-hits.
These rows are not yet in `mac1_splits.csv`; add them to the cluster-based split before using them for evaluation.

## Ligand codes (table SMILES vs. mmCIF ligands)

`experiments/match_mac1_ligands.py` writes `data/processed/mac1_ligands.csv`: for each of the 182 IC50 pairs it compares the
InChIKey of the table SMILES (largest fragment) with the InChIKey of every non-polymer component in that structure's mmCIF
(`_pdbx_entity_nonpoly`, waters dropped), using the RCSB chemical component CIFs saved in `data/raw/ccd/` (191 codes are not in Boltz's CCD cache).

- 88 pairs match on connectivity and stereochemistry, 92 on connectivity only, 2 do not match. Every pair resolves to a ligand code
  in its own structure, so the table SMILES can be used to pick the affinity ligand among DMS/CL/TFA/glycerol/etc.
- 29 pairs keep two candidate codes: the crystal models both enantiomers of a racemic compound (e.g. QW3/QWC in 5SQW, A1CJX/A1CJY in 7IJ2).
  The IC50 is for the mixture; pick one enantiomer (or both, as a separate entry) when building records.
- Connectivity-only matches (92) are mostly racemic or unspecified stereo in the table (most have undefined centres) or drawn as the other
  enantiomer (e.g. 14AB, 5SOJ, 7HE8, 7HF7, 7HFR, 9D6G). The `model_smiles` column holds the isomeric SMILES re-derived from the deposited coordinates; it agrees with the component's stereo for every matched pair, so that is the one to trust for the pose. In 5SRM the two RBB protomers model opposite stereo (`model_copies_consistent` is False).
- **Mismatches:** 7HFU (AVI-6002) and 7HFX (AVI-5266). The deposited ligand (A1BBX, A1BCC) is a different compound from the table SMILES.
  The IC50 may belong to the other compound or the table is mislabelled. Both are in the val/test splits; exclude from affinity training/eval until resolved.
- 96 pairs have a ligand code absent from Boltz's CCD cache, so a molecule must be built from the component CIF before `parse_mmcif`.

### Non-hit training records — `data/processed/mac1_nonhit_records.csv`

Rebuild: `~/venvs/boltz2/bin/python experiments/build_mac1_nonhit_records.py`. One row per negative ligand (2347: 2316 `xtal_nonhit`, `weak`; 31 `docked_unconfirmed_htrf` with < 30% HTRF displacement at the top concentration, `weak_htrf`; the 6 docked compounds at >= 50% and the 2 without an HTRF value are not used; salt forms are matched on the largest fragment, which removed 9 salt duplicates and the histamine entry that is the 5RUS hit), in the
column style of the other tables (see `data_schema.md`): `sequence` (169-residue Mac1, asserted equal to chain A of the structure),
`smiles`, `pdb_id` = `7KQO` and `chain` = `A` (apo P43; the structure is `data/structures/mac1/7KQO.cif`, `diffusion_train` True),
`split` = train, `structure_loss_weight` = 0 (no ligand pose exists), `affinity_binary` = 0, `label_quality` = weak.
Chain A because fragments bind protomer A of the P43 crystals (Data S1 sheet 11: 179 of 204 adenosine-site hits are on protomer A);
the adenosine site of protomer B is blocked by crystal packing, so a non-hit there is uninformative.
Leakage: the script drops any non-hit with Tanimoto >= 0.5 (Morgan r=2, 2048 bit) to a val/test ligand; none were that close (max 0.45).
Not yet converted to Boltz `StructureV2`/`Record`/`manifest.json` (those do not exist yet, see `data_schema.md`); the conversion step
must carry `structure_loss_weight`, `affinity_binary` and the ligand SMILES with no ligand chain in the structure.
Training caveat: all rows share one protein, one structure and one label, so on their own they teach only "this ligand is a non-binder".
They are useful only mixed with the IC50 positives, and these fragments are small (median 13 heavy atoms vs 20 for the AVI series),
so a model can separate classes by size alone; check that before trusting a binder-probability gain. Weight them low: soak non-hits are weak negatives.

### Fragment-hit records and the Schuller IC50s — `data/processed/mac1_fragment_hit_records.csv` (2026-10-14)

Rebuild: `PYTHONPATH=src ~/venvs/boltz2/bin/python experiments/build_mac1_fragment_hit_records.py`. The fragment-hit crystal structures were
already among the 603 mmCIFs (232 of the 236 hit entries in Data S1; they trained the diffusion head only). This table adds labels:
240 records from 229 entries, one per (entry, protomer) row of sheet 11, each with its real ligand pose. The 3 entries without a downloaded
structure (5S39, 5S3F, 5S4E) are skipped; 5S6W (sheet 12) is also not in the downloaded set.

- `structure_loss_weight` 1. `affinity_binary` 1 only for adenosine-site hits (183); surface, K90, catalytic-loop and crystal-contact hits (57) are real poses
  but not evidence of binding at the active site, so their binary label is empty.
- `smiles` comes from the deposited component, so it matches the pose; `smiles_match` compares it with the screen's library SMILES (216 stereo, 21 connectivity,
  3 none: 5S1U, 5S40 and 5RSI differ, e.g. non-aromatic pyridone tautomers in the CCD component).
- `sequence` is the entry's own polymer: the 13 C2 entries use a 173-residue construct (SNAGEV...), the 227 P43 records the 169-residue one.
- `split` is `held_out_similar` for 2 hits (5RV6, 5RVD) that have Tanimoto >= 0.5 to a val/test ligand: leave them out of training.
- Caveat: the fragments are still hits at 0.1-1 M soaks, so these are crystal-binding positives, not potency labels. 8 carry a measured IC50 (183-5500 µM).
  5S3W's IC50 belongs to a catalytic-site protomer-B hit, so its `affinity_binary` is empty while `ic50_uM` is set.

`schuller2021` IC50s (Data S1 sheet 12, 8 fragments with a numeric HTRF IC50; the `~5500` for SF048 is approximate, its `=` qualifier is not exact)
were added to `mac1_affinity.csv` by `build_mac1_dataset.py` (needs the local xlsx; the script was not run end to end here because it needs network, so the
rows were appended with the same function). Splits were regenerated with seed 0: 171 pairs (train 125, val 21, test 25).
**Existing splits changed**: adding the 8 rows shifts the cluster and target sizes, so 39 of the previous 163 pairs changed split (18 train -> val, 20 val -> train, 1 val -> test);
all 24 previous test pairs stayed in test, and no pre-cutoff pair moved to val/test. Any result pinned to the earlier table (115/24/24) is not comparable. The earlier files are not kept in the repo.

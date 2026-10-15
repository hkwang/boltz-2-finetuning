# Boltz-2 fine-tuning

Research code for fine-tuning Boltz-2 on protein–ligand structure and affinity
pairs. Keep the repository organized by responsibility as the implementation
grows:

```text
configs/                 experiment and data configuration
src/boltz2_finetuning/   reusable data, training, and evaluation code
experiments/             experiment-specific entry points and run scripts
jobs/polaris/            Polaris PBS submission templates
testing/                 local smoke checks and dummy jobs
docs/                    shared notes; data_schema.md (table layout), datasets.md, mac1_cofold.md, regression_check.md, boltz2_polaris_log.md
data/                    raw/ supplements and processed/ tables
```

Only directories with actual content are created. ALCF PBS usage and
configuration instructions are in [`jobs/polaris/README.md`](jobs/polaris/README.md).
The local wrapper check is `bash testing/test_pbs_scripts.sh`.

Storage: `/home` has a 45 GB quota. `experiments/results` and `~/.cache/boltz` are
symlinks into `/eagle/GenesisHackathonOct26/hkwang/boltz2-finetuning/`; jobs must
request `-l filesystems=home:eagle`. Details in
[`docs/diffusion_finetuning_progress.md`](docs/diffusion_finetuning_progress.md).

"""Build Boltz CCD-format molecules for every Mac1 ligand code that is missing from the Boltz CCD cache.

Reads data/raw/ccd/{CODE}.cif (fetched by match_mac1_ligands.py) and writes data/processed/mac1_mols/{CODE}.pkl.
The shared ~/.cache/boltz/mols is never modified; pass the output directory as an extra molecule directory.

Run: PYTHONPATH=src python experiments/build_mac1_mols.py
"""

import argparse
import pickle
from pathlib import Path

from boltz2_finetuning.ccd import build_ccd_mol

CCD_CACHE = Path.home() / ".cache" / "boltz" / "mols"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--component-dir", type=Path, default=Path("data/raw/ccd"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/mac1_mols"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    missing = [path for path in sorted(args.component_dir.glob("*.cif")) if not (CCD_CACHE / f"{path.stem}.pkl").exists()]
    failed = []
    for path in missing:
        try:
            mol = build_ccd_mol(path.stem, path, args.seed)
        except Exception as error:
            failed.append((path.stem, repr(error)))
            continue
        with (args.output_dir / f"{path.stem}.pkl").open("wb") as handle:
            pickle.dump(mol, handle)
    print(f"built {len(missing) - len(failed)}/{len(missing)} molecules -> {args.output_dir}")
    for code, error in failed:
        print("FAILED", code, error)


if __name__ == "__main__":
    main()

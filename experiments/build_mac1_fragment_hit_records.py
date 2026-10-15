"""Format the Mac1 fragment-screen hits (Schuller et al., Sci. Adv. 2021) as training records with a real ligand pose.

One record per (PDB entry, protomer) row of Data S1 sheet "11. fragment_hits_classified" whose entry is in
data/processed/mac1_structures.csv. The ligand is the non-additive het residue in that chain of the deposited mmCIF
(DMSO, glycerol and ions are ignored); `smiles` is generated from that component (Boltz CCD cache), so it matches the
modelled ligand, and `smiles_match` says how it compares to the screen's library SMILES (stereo/connectivity/none).
`sequence` is the entry's own polymer sequence: the 13 C2 entries use a 173-residue construct, the P43 entries the
169-residue one.

Labels: `structure_loss_weight` = 1 (the pose is real). `affinity_binary` = 1 only for adenosine-site hits
(`general_site` == Adenosine); surface, K90, catalytic-loop and crystal-contact hits are real poses but not evidence of
active-site binding, so their `affinity_binary` is empty (no affinity supervision). Entries with a measured IC50 in
mac1_splits.csv carry it in `ic50_uM`. `split` is train, or held_out_similar when the ligand has Tanimoto >= 0.5 to a
val/test ligand (leave those out of training). Hit entries without a downloaded structure (5S39, 5S3F, 5S4E) are skipped.

Run from the repository root: PYTHONPATH=src python experiments/build_mac1_fragment_hit_records.py
"""

import argparse
import pickle
from pathlib import Path

import gemmi
import pandas as pd
from rdkit import Chem, RDLogger

from boltz2_finetuning.ligands import match_level, smiles_inchikey
from boltz2_finetuning.splits import close_to_held_out

RDLogger.DisableLog("rdApp.*")
SUPP = "abf8711_data_file_s1.xlsx"
CCD_CACHE = Path.home() / ".cache" / "boltz" / "mols"
ADDITIVES = {"HOH", "DMS", "EDO", "GOL", "SO4", "CL", "NA", "PO4", "ACT", "MG", "K", "ZN", "PEG", "PG4", "FMT", "NO3", "CA"}
HELD_OUT_SIMILARITY = 0.5
ACTIVE_SITE = "Adenosine"


def entry_sequence(block: gemmi.cif.Block) -> str:
    sequences = [gemmi.cif.as_string(value).replace("\n", "") for value in block.find_values("_entity_poly.pdbx_seq_one_letter_code_can")]
    assert len(sequences) == 1, "expected a single polymer entity"
    return sequences[0]


def ligand_smiles(code: str) -> str:
    mol = pickle.load(open(CCD_CACHE / f"{code}.pkl", "rb"))
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


def chain_ligand(structure: gemmi.Structure, chain_name: str, library_smiles: str) -> tuple[str, str] | None:
    """(component code, match level to the library SMILES) of the best-matching ligand in the chain, or None."""
    chain = next((chain for chain in structure[0] if chain.name == chain_name), None)
    codes = sorted({residue.name for residue in chain or [] if residue.het_flag == "H" and residue.name not in ADDITIVES})
    if not codes:
        return None
    library_key = smiles_inchikey(library_smiles)
    levels = {code: match_level(library_key, smiles_inchikey(ligand_smiles(code))) for code in codes}
    best = min(codes, key=lambda code: ("stereo", "connectivity", "none").index(levels[code]))
    return best, levels[best]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/mac1"))
    parser.add_argument("--structure-dir", type=Path, default=Path("data/structures/mac1"))
    parser.add_argument("--structures", type=Path, default=Path("data/processed/mac1_structures.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/mac1_splits.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/mac1_fragment_hit_records.csv"))
    args = parser.parse_args()

    hits = pd.read_excel(args.raw_dir / SUPP, sheet_name="11. fragment_hits_classified", header=1)
    available = set(pd.read_csv(args.structures)["pdb_id"])
    skipped = sorted(set(hits["PDB"].dropna()) - available)
    hits = hits[hits["PDB"].isin(available)]
    splits = pd.read_csv(args.splits)
    ic50 = splits.set_index("pdb_id")["ic50_uM"]

    rows = []
    for hit in hits.itertuples():
        chain_name = hit.Model[-1] if hit.Model.endswith(("_A", "_B")) else "A"
        path = args.structure_dir / f"{hit.PDB}.cif"
        ligand = chain_ligand(gemmi.read_structure(str(path)), chain_name, hit.Smiles)
        if ligand is None:
            continue
        code, level = ligand
        rows.append({
            "record_id": f"mac1_hit_{hit.PDB}_{chain_name}",
            "sequence": entry_sequence(gemmi.cif.read(str(path))[0]),
            "smiles": ligand_smiles(code),
            "pdb_id": hit.PDB,
            "chain": chain_name,
            "ligand_code": code,
            "smiles_match": level,
            "site": hit.General_site,
            "crystal_form": hit.Crystals,
            "structure_loss_weight": 1.0,
            "affinity_binary": 1 if hit.General_site == ACTIVE_SITE else pd.NA,
            "ic50_uM": ic50.get(hit.PDB, pd.NA),
            "label_quality": "xtal_hit",
            "compound_id": hit.Compound_ID,
            "library": hit.Library,
            "source": "abf8711_data_file_s1:sheet 11",
        })
    records = pd.DataFrame(rows)
    close = close_to_held_out(records["smiles"], splits, HELD_OUT_SIMILARITY)
    records.insert(5, "split", close.map({False: "train", True: "held_out_similar"}))
    records.to_csv(args.output, index=False)

    print(f"skipped entries without a downloaded structure: {skipped}")
    print(f"wrote {len(records)} records ({records['pdb_id'].nunique()} entries) to {args.output}")
    print(records.groupby(["site", "split"]).size().to_string())
    print("smiles_match:", records["smiles_match"].value_counts().to_dict())
    print("with measured IC50:", int(records["ic50_uM"].notna().sum()))


if __name__ == "__main__":
    main()

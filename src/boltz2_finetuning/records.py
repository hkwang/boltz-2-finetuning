"""Write Mac1 structures and SMILES as Boltz-2 training inputs.

Layout under the output directory (the files `inferencev2.load_input` reads, plus a manifest):
  structures/{id}.npz   StructureV2          constraints/{id}.npz  ResidueConstraints
  mols/{id}.pkl         extra RDKit mols     records/{id}.json     Record (manifest.json lists all of them)
Proteins use `msa_id = -1` (single sequence), as in the zero-shot co-folding runs.
"""

import pickle
from pathlib import Path

import gemmi
from boltz.data import const
from boltz.data.parse.mmcif_with_constraints import parse_mmcif
from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.types import AffinityInfo, ChainInfo, InterfaceInfo, Record, StructureV2
from rdkit import Chem
from rdkit.Chem import Descriptors

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
NONPOLYMER = const.chain_type_ids["NONPOLYMER"]
MAX_AFFINITY_ATOMS = 128


def find_ligand_chain(structure_path: Path, structure: StructureV2, code: str, author_chain: str | None) -> int | None:
    """asym_id of the first parsed chain holding ligand `code` (in author chain `author_chain` if given)."""
    deposited = gemmi.read_structure(str(structure_path))
    for chain in deposited[0]:
        if author_chain is not None and chain.name != author_chain:
            continue
        for residue in chain:
            if residue.name != code:
                continue
            for parsed in structure.chains:
                if parsed["mol_type"] == NONPOLYMER and str(parsed["name"]) == residue.subchain:
                    return int(parsed["asym_id"])
    return None


def build_record(record_id: str, info, structure: StructureV2, affinity: AffinityInfo | None) -> Record:
    chains = [
        ChainInfo(
            chain_id=int(chain["asym_id"]),
            chain_name=str(chain["name"]),
            mol_type=int(chain["mol_type"]),
            cluster_id=-1,
            msa_id=-1,
            num_residues=int(chain["res_num"]),
            valid=True,
            entity_id=int(chain["entity_id"]),
        )
        for chain in structure.chains
    ]
    interfaces = [InterfaceInfo(int(row["chain_1"]), int(row["chain_2"])) for row in structure.interfaces]
    return Record(id=record_id, structure=info, chains=chains, interfaces=interfaces, affinity=affinity)


def convert_crystal(
    pdb_id: str,
    structure_path: Path,
    built_mols: dict[str, Chem.Mol],
    mol_dir: Path,
    affinity_code: str | None = None,
    author_chain: str | None = None,
):
    """Parse one mmCIF (asymmetric unit) into (record, structure, constraints, extra_mols).

    The asymmetric unit is used because Mac1 is monomeric and assembly 1 can omit the modelled ligand or
    protomer (e.g. 5S3W). `affinity_code` marks one copy of that ligand as the affinity ligand.
    """
    mols = dict(built_mols)
    parsed = parse_mmcif(str(structure_path), mols=mols, moldir=str(mol_dir), use_assembly=False)
    affinity = None
    if affinity_code is not None:
        chain_id = find_ligand_chain(structure_path, parsed.data, affinity_code, author_chain)
        if chain_id is None:
            raise ValueError(f"{pdb_id}: ligand {affinity_code} not found in chain {author_chain}")
        heavy_atoms = mols[affinity_code].GetNumAtoms()
        assert heavy_atoms <= MAX_AFFINITY_ATOMS, f"{pdb_id}: {affinity_code} has {heavy_atoms} atoms"
        affinity = AffinityInfo(chain_id=chain_id, mw=Descriptors.MolWt(mols[affinity_code]))
    used = {str(residue["name"]) for residue in parsed.data.residues}
    extra_mols = {code: mol for code, mol in built_mols.items() if code in used}
    record = build_record(pdb_id, parsed.info, parsed.data, affinity)
    return record, parsed.data, parsed.residue_constraints, extra_mols


def convert_smiles(record_id: str, sequence: str, smiles: str, canonicals: dict, mol_dir: Path):
    """Boltz's own sequence + SMILES path (single-sequence MSA, ligand conformer, no pose) with the ligand as affinity target."""
    schema = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": sequence, "msa": "empty"}},
            {"ligand": {"id": "B", "smiles": smiles}},
        ],
        "properties": [{"affinity": {"binder": "B"}}],
    }
    target = parse_boltz_schema(record_id, schema, canonicals, mol_dir, boltz_2=True)
    return target.record, target.structure, target.residue_constraints, target.extra_mols


def write_inputs(out_dir: Path, record: Record, structure: StructureV2, constraints, extra_mols: dict) -> None:
    for name in ("structures", "constraints", "mols", "records"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    structure.dump(out_dir / "structures" / f"{record.id}.npz")
    constraints.dump(out_dir / "constraints" / f"{record.id}.npz")
    with (out_dir / "mols" / f"{record.id}.pkl").open("wb") as handle:
        pickle.dump(extra_mols, handle)
    record.dump(out_dir / "records" / f"{record.id}.json")

"""Cross-reference table SMILES against the non-polymer components deposited in each mmCIF."""

import ssl
import urllib.request
from pathlib import Path

import certifi
import gemmi
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi

RCSB_COMPONENT_URL = "https://files.rcsb.org/ligands/download/{code}.cif"
IGNORED_CODES = {"HOH"}
CONNECTIVITY_CHARS = 14
STEREO_END_CHAR = 25

RDLogger.DisableLog("rdApp.*")


def nonpolymer_codes(structure_path: Path) -> list[str]:
    """Component codes of all non-polymer entities (waters excluded), in file order."""
    block = gemmi.cif.read(str(structure_path))[0]
    codes = [row[0] for row in block.find("_pdbx_entity_nonpoly.", ["comp_id"])]
    return [code for code in codes if code not in IGNORED_CODES]


def download_component(code: str, component_dir: Path) -> Path:
    """RCSB chemical component CIF; Boltz's CCD cache lacks the 5-character A1xxx codes of recent Mac1 entries."""
    target = component_dir / f"{code}.cif"
    if not target.exists():
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(RCSB_COMPONENT_URL.format(code=code), context=context, timeout=60) as response:
            target.write_bytes(response.read())
    return target


def component_inchikey(component_path: Path) -> str | None:
    block = gemmi.cif.read(str(component_path))[0]
    for row in block.find("_pdbx_chem_comp_descriptor.", ["type", "descriptor"]):
        if row[0] == "InChIKey":
            return gemmi.cif.as_string(row[1])
    return None


def smiles_inchikey(smiles: str) -> str | None:
    """InChIKey of the largest fragment (drops counter-ions such as HCl); None if RDKit cannot parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    largest = max(Chem.GetMolFrags(mol, asMols=True), key=lambda fragment: fragment.GetNumHeavyAtoms())
    return inchi.MolToInchiKey(largest)


def match_level(table_key: str, component_key: str) -> str:
    """'stereo' if connectivity and stereo layers agree, 'connectivity' if only the skeleton does, else 'none'."""
    if table_key[:CONNECTIVITY_CHARS] != component_key[:CONNECTIVITY_CHARS]:
        return "none"
    if table_key[CONNECTIVITY_CHARS:STEREO_END_CHAR] == component_key[CONNECTIVITY_CHARS:STEREO_END_CHAR]:
        return "stereo"
    return "connectivity"


def assign_ligand(table_key: str, component_keys: dict[str, str]) -> tuple[list[str], str]:
    """Best-matching component codes (all ties kept) and the match level they share."""
    levels = {code: match_level(table_key, key) for code, key in component_keys.items()}
    for level in ("stereo", "connectivity"):
        codes = [code for code, code_level in levels.items() if code_level == level]
        if codes:
            return codes, level
    return [], "none"


def model_smiles(structure_path: Path, code: str, mol: Chem.Mol) -> tuple[str | None, bool]:
    """Isomeric SMILES of the ligand as modelled in the structure, plus whether all complete copies agree.

    `mol` is the Boltz-format component (heavy atoms, `name` atom property). Stereo is reassigned from the deposited
    coordinates of each fully modelled copy; the first copy is returned. (None, True) if no copy is fully modelled.
    """
    structure = gemmi.read_structure(str(structure_path))
    structure.remove_hydrogens()
    structure.remove_alternative_conformations()
    names = [atom.GetProp("name") for atom in mol.GetAtoms()]
    found = []
    for chain in structure[0]:
        for residue in chain:
            if residue.name != code:
                continue
            positions = {atom.name: atom.pos for atom in residue}
            if any(name not in positions for name in names):
                continue
            modelled = Chem.Mol(mol)
            modelled.RemoveAllConformers()
            conformer = Chem.Conformer(modelled.GetNumAtoms())
            for index, name in enumerate(names):
                conformer.SetAtomPosition(index, (positions[name].x, positions[name].y, positions[name].z))
            conformer.Set3D(True)
            modelled.AddConformer(conformer)
            for atom in modelled.GetAtoms():
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            Chem.AssignStereochemistryFrom3D(modelled)
            found.append(Chem.MolToSmiles(modelled))
    return (found[0], len(set(found)) == 1) if found else (None, True)

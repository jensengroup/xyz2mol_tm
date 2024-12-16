from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# The tmQMg data is required for the code
__current__ = Path(__file__).parent.absolute()
PATH_TO_TMQMG = __current__ / Path("../tmQMg")

# Check if the folder exists
if not PATH_TO_TMQMG.exists():
    raise FileNotFoundError(f"The tmQMg data does not exist at {PATH_TO_TMQMG}.")

# Units for the various targets used in ML
TARGET_UNITS = {
    "tzvp_homo_lumo_gap": "mHa",
    "polarisability": "bohr³",
    "tzvp_dipole_moment": "D",
}

from xyz2mol_tm.NBO_to_smiles.mol_utils import TRANSITION_METALS

tm = f"[{','.join(TRANSITION_METALS)}]"


# RDKit chiral tag mapper
chiral_tags = {
    "CHI_ALLENE": Chem.CHI_ALLENE,
    "CHI_OCTAHEDRAL": Chem.CHI_OCTAHEDRAL,
    "CHI_OTHER": Chem.CHI_OTHER,
    "CHI_SQUAREPLANAR": Chem.CHI_SQUAREPLANAR,
    "CHI_TETRAHEDRAL": Chem.CHI_TETRAHEDRAL,
    "CHI_TETRAHEDRAL_CCW": Chem.CHI_TETRAHEDRAL_CCW,
    "CHI_TETRAHEDRAL_CW": Chem.CHI_TETRAHEDRAL_CW,
    "CHI_TRIGONALBIPYRAMIDAL": Chem.CHI_TRIGONALBIPYRAMIDAL,
    "CHI_UNSPECIFIED": Chem.CHI_UNSPECIFIED,
}


def mae(predictions, targets):
    errors = np.abs(predictions - targets)
    return np.mean(errors)


def median(predictions, targets):
    errors = np.abs(predictions - targets)
    return np.median(errors)


def rmse(predictions, targets):
    errors = np.abs(predictions - targets)
    return np.sqrt(np.mean(np.power(errors, 2)))


def r_squared(predictions, targets):
    target_mean = np.mean(targets)
    return 1 - (
        np.sum(np.power(targets - predictions, 2))
        / np.sum(np.power(targets - target_mean, 2))
    )


def setChiralTagAndOrder(atom, chiralTag, chiralPermutation=None):
    """Sets the chiral tag of an atom and the permutation order of attached
    ligands. These tags are used in RDKit embedding into 3D structures.

    Args:
    atom (Chem.Atom): Atom for which to set the chiral tag/permutation order properties
    chiralTag (Chem.rdchem.ChiralType, optional): Chiral Tag of Metal Atom.
    permutationOrder (int, optional): Permutation order of ligands.
    """
    atom.SetChiralTag(chiralTag)
    if chiralPermutation:
        atom.SetIntProp("_chiralPermutation", chiralPermutation)


def set_chiral_tag_on_smiles(row, smiles_column_string):
    m = Chem.MolFromSmiles(row[smiles_column_string])

    metal_atom_idx = m.GetSubstructMatches(Chem.MolFromSmarts(tm))[0][0]

    metal_atom = m.GetAtomWithIdx(metal_atom_idx)

    if row["chiralPermutation"] != row["chiralPermutation"]:
        chiralPermutation = None
    else:
        chiralPermutation = int(row["chiralPermutation"])

    setChiralTagAndOrder(
        atom=metal_atom,
        chiralTag=chiral_tags[row["chiralTag"]],
        chiralPermutation=chiralPermutation,
    )

    chiral_smiles = Chem.MolToSmiles(m)

    return chiral_smiles


def smiles2fp(smiles, encoding="count", includeChirality=False):
    """Convert smiles into a morgan fingerprint in count or bit form."""
    mol = Chem.MolFromSmiles(smiles)
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=1024, includeChirality=includeChirality
    )
    if encoding == "count":
        fp = fp_gen.GetCountFingerprintAsNumPy(mol)
    elif encoding == "one-hot":
        fp = fp_gen.GetFingerprintAsNumPy(mol)
    return fp


def get_dataset_path(dataset):
    if dataset == "tmqmg":
        dataset_path = Path("../SMILES_csvs/tmqmg_smiles.csv")
    elif dataset == "csd":
        dataset_path = Path(
            "../SMILES_csvs/260k_csd_data_with_chiraltags_and_fixedsmiles.csv"
        )
    else:
        raise Exception("Not valid path")

    return dataset_path


def map_smiles_column_name(chosen_smiles):
    "Map a chosen SMILES set into the corresponding dataframe column header"
    if "huckel" in chosen_smiles:
        smiles_column_string = "smiles_huckel_DFT_xyz"
    elif "tmqmg" in chosen_smiles:
        smiles_column_string = "smiles_NBO_DFT_xyz"
    elif "csd_fixed_smiles" in chosen_smiles:
        smiles_column_string = "smiles_CSD_fix"
    else:
        raise Exception("Not valid smiles")
    return smiles_column_string

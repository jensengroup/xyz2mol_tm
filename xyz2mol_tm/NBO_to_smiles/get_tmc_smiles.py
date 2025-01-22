"""Driver module for getting the TMC SMILES from the tmQMg dataset based on the
data in tmQMg-L."""

import argparse
import ast
import logging
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import data_handler
import mol_utils
import pandas as pd
import tqdm
from data_handler import get_connection_ids, get_stable_occ
from mol_utils import (
    attach_ligands,
    combine_molecules,
    get_git_revision_short_hash,
    get_transition_metal_atom,
    process_paralell,
    set_metal_coordinates,
    xyz_string_decompose,
)
from rdkit import Chem
from rdkit.Geometry import Point3D

from xyz2mol_tm import xyz2mol_local, xyz2mol_tmc

ALLOWED_OXIDATION_STATES, TRANSITION_METALS = (
    mol_utils.ALLOWED_OXIDATION_STATES,
    mol_utils.TRANSITION_METALS,
)

_logger: logging.Logger = logging.getLogger("tmc_smiles")


__location__ = Path(__file__).parent.absolute()


def get_ligand_data(complex_ligand_composition, complex_id):
    """Get ligand mol objects and coordination atoms.

    Args:
        complex_ligand_composition (List(tuple)): Defines the ligands present in a TMC
        complex_id (str): The tmQMg TMC identifier

    Returns:
        ligand_data (dict): A dictionary containing the connection atoms and mol objects for each ligand defined in the ligand composition
    """
    ligand_data = defaultdict(dict)

    for ligand, ligand_subgraph in complex_ligand_composition:
        try:
            ligand_row = df2.loc[ligand]
            if ligand_row.empty:
                _logger.warning(f"Ligand {ligand} not found in data.")
                return None

            ligand_result = get_smiles_for_tmqmg_ligand(
                ligand_row, ligand_subgraph, most_stable=False
            )
            if ligand_result:
                dict_entry = ligand + "_" + ligand_subgraph
                mol, connection_ids = ligand_result
                ligand_data[dict_entry] = {
                    "connection_atoms": connection_ids,
                    "mol": mol,
                }
            else:
                _logger.warning(
                    f"Failed to generate mol object for {ligand} with subgraph {ligand_subgraph}"
                )
                dict_entry = ligand + "_" + ligand_subgraph
                ligand_data[dict_entry] = None

        except KeyError:
            _logger.error(f"Ligand {ligand} missing in df2.")
            return None

    if any(value is None for value in ligand_data.values()):
        _logger.error(f"Complex {complex_id} did not get mol objects for all ligands")
        return None

    return ligand_data


def construct_tmc_mol(ligand_data, complex_id):
    """Combine the ligand mol objects with a TM atom to create the final TMC
    mol object.

    Args:
        ligand_data (dict): Contains the ligand mol objects and coordination atom indices
        complex_id (str): tmQMg complex identifier

    Returns:
        tmc_mol (rdkit.Chem.Mol): The TMC mol object
    """
    # Get the transition metal atom and number of atoms in the complex
    tm_atom, atom_count = get_transition_metal_atom(complex_id)
    if not tm_atom:
        _logger.warning(f"Transition metal not found in {complex_id}")
        return None

    _logger.debug("Combine fragments into one mol")
    metal_mol = Chem.MolFromSmarts(f"[{tm_atom}]")
    combined_mol = combine_molecules(metal_mol, ligand_data)

    _logger.debug("Attaching the fragments in the mol to the TM")
    tmc_mol = attach_ligands(combined_mol, ligand_data)

    if tmc_mol.GetNumAtoms() != atom_count:
        _logger.error(
            f"Number of atoms in the obtained mol object does not match the number of atoms in the TMC xyz file for {complex_id}"
        )
        return None
    _logger.debug(f"Succesfully constructed the TMC mol for {complex_id}")

    # Get charge for the transition metal
    complex_charge = tmQMg_properties[tmQMg_properties["id"] == complex_id][
        "charge"
    ].item()

    # Get total ligand charge
    ligand_charges = []
    for ligand, res in ligand_data.items():
        charge = int(df2.loc[ligand.split("_")[0]]["charge"])
        ligand_charges.append(charge)
    total_ligand_charge = sum(ligand_charges)
    metal_charge = complex_charge - total_ligand_charge

    metal_idx = tmc_mol.GetSubstructMatch(metal_mol)[0]
    metal_atom = tmc_mol.GetAtomWithIdx(metal_idx)
    metal_atom.SetFormalCharge(metal_charge)

    if metal_charge not in ALLOWED_OXIDATION_STATES.get(metal_atom.GetSymbol(), []):
        _logger.warning(
            f"Questionable oxidation state for metal {metal_atom.GetSymbol()} in complex {complex_id}"
        )
    _logger.info("Setting metal coordinates")

    set_metal_coordinates(tmc_mol, metal_idx, complex_id)

    # Get potential chiral tags. This was not used as RDKit functionality on this is currently unpredictable.
    # haptic_mol = Chem.DativeBondsToHaptic(tmc_mol)
    # Chem.SanitizeMol(haptic_mol)
    # Chem.AssignStereochemistry(haptic_mol)
    # Chem.AssignAtomChiralTagsFromStructure(haptic_mol)
    # Chem.AssignStereochemistryFrom3D(haptic_mol)
    # chiralTag = haptic_mol.GetAtomWithIdx(metal_idx).GetChiralTag()
    # tmc_mol = Chem.HapticBondsToDative(haptic_mol)

    # Put chiral tags on the metal center
    xyz2mol_local.chiral_stereo_check(tmc_mol)

    return tmc_mol


def process_complex(complex_tuple):
    """Process a transition metal complex to generate TMC SMILES strings and
    related data.

    Args:
        complex_label (str): The tmQMg identifier of the transition metal complex.
    Returns:
        tuple: (complex_result_dict, complex_id)
    """
    # print(complex_tuple)
    complex_id, complex_ligand_composition = complex_tuple
    _logger.info(f"Processing complex {complex_id}")

    # Return early if no ligands are provided
    if not complex_ligand_composition:
        _logger.info(f"{complex_id} has no ligands")
        return None, complex_id

    # Process ligands and generate molecular data
    ligand_data = get_ligand_data(complex_ligand_composition, complex_id)
    if ligand_data is None:
        return None, complex_id

    tmc_mol = construct_tmc_mol(ligand_data, complex_id)
    if not tmc_mol:
        return None, complex_id

    # Sanitize the molecule
    try:
        Chem.SanitizeMol(tmc_mol)
        complex_smi = Chem.MolToSmiles(Chem.RemoveHs(tmc_mol))
    except Exception as e:
        _logger.error(f"Sanitization failed for complex {complex_id} with message: {e}")
        return None, complex_id

    # Perform 2 TMC smiles fixed
    try:
        complex_smi = xyz2mol_tmc.fix_equivalent_Os(complex_smi)
        complex_smi = xyz2mol_tmc.fix_NO2(complex_smi)
    except Exception as e:
        _logger.error(f"Fixes failed for complex {complex_id} with message: {e}")

    # Verify SMILES
    try:
        Chem.MolFromSmiles(complex_smi)
    except Exception:
        _logger.error(f"Invalid SMILES string for complex {complex_id}")
        return None, complex_id

    # Gather results
    ligand_result = {
        ligand_subgraph: Chem.MolToSmiles(data["mol"])
        for ligand_subgraph, data in ligand_data.items()
    }

    complex_result_dict = {
        "complex_id": complex_id,
        "complex_smi": complex_smi,
        "ligand_smiles": ligand_result,
    }

    _logger.info(f"Successfully processed complex {complex_id}")
    return complex_result_dict, complex_id


def get_lig_mol_fixed_charge(xyz, charge, coordinating_atoms):
    """Get the mol object for a ligand defined by an xyz string, charge and
    coordination atoms.

    Args:
        xyz (str): xyz coordinates in one long string
        charge (int): charge of ligand
        coordinating_atoms (List(int)): List of coorindation atom indices.

    Returns:
        mol (rdkit.Chem.rdchem.Mol): mol object
    """
    if len(xyz.split("\n")) < 4:
        _logger.info(f"Using small scorer for ligand {xyz}")
        mol = Chem.MolFromSmiles("[H]")
        mol.GetAtomWithIdx(0).SetFormalCharge(charge)
        mol.UpdatePropertyCache()
        mol.GetAtomWithIdx(0).SetNumRadicalElectrons(0)
        conf = Chem.Conformer()
        line = xyz.split("\n")[-1]
        splits = line.split(" ")
        xyz_coords = [float(x) for x in splits[1:4]]
        conf.SetAtomPosition(0, Point3D(xyz_coords[0], xyz_coords[1], xyz_coords[2]))
        mol.AddConformer(conf)
        lig_mol = mol
    else:
        atoms, xyz_coords = xyz_string_decompose(xyz)
        AC, proto_mol = xyz2mol_local.xyz2AC_obabel(atoms, xyz_coords)
        lig_mol = xyz2mol_local.AC2mol(
            proto_mol,
            AC,
            atoms,
            charge,
            allow_charged_fragments=True,
            use_atom_maps=False,
        )
    # print("first:", Chem.MolToSmiles(lig_mol))
    possible_res_mols = xyz2mol_tmc.lig_checks(lig_mol, coordinating_atoms)
    # print("charge =", charge)
    best_res_mol, lowest_pos, lowest_neg, highest_aromatic = possible_res_mols[0]
    for res_mol, N_pos_atoms, N_neg_atoms, N_aromatic in possible_res_mols:
        if N_aromatic > highest_aromatic:
            best_res_mol, lowest_pos, lowest_neg, highest_aromatic = (
                res_mol,
                N_pos_atoms,
                N_neg_atoms,
                N_aromatic,
            )
        if (
            N_aromatic == highest_aromatic
            and N_pos_atoms + N_neg_atoms < lowest_pos + lowest_neg
        ):
            best_res_mol, lowest_pos, lowest_neg = res_mol, N_pos_atoms, N_neg_atoms

    if lowest_pos + lowest_neg == 0:
        # print("found opt solution")
        return best_res_mol

    lig_mol_no_carbene = xyz2mol_local.AC2mol(
        proto_mol,
        AC,
        atoms,
        charge,
        allow_charged_fragments=True,
        use_atom_maps=False,
        allow_carbenes=False,
    )

    if lig_mol_no_carbene:
        # print("found solution with no carbene, charge =", charge)
        res_mols_no_carbenes = xyz2mol_tmc.lig_checks(
            lig_mol_no_carbene, coordinating_atoms
        )
        for res_mol, N_pos_atoms, N_neg_atoms, N_aromatic in res_mols_no_carbenes:
            if (
                N_aromatic > highest_aromatic
                and N_pos_atoms + N_neg_atoms <= lowest_pos + lowest_neg
            ):
                best_res_mol, lowest_pos, lowest_neg, highest_aromatic = (
                    res_mol,
                    N_pos_atoms,
                    N_neg_atoms,
                    N_aromatic,
                )
            if (
                N_aromatic == highest_aromatic
                and N_pos_atoms + N_neg_atoms < lowest_pos + lowest_neg
            ):
                best_res_mol, lowest_pos, lowest_neg = res_mol, N_pos_atoms, N_neg_atoms

    return best_res_mol


def get_smiles_for_tmqmg_ligand(row, ligands_xyz_file, most_stable=False):
    """Function used to obtain ligand mol objects. The input is a row from a
    dataframe of the tmQMg-L dataset.

    Args:
        row (tmQMg-L row): A row from the tmQMg-L dataframe
        ligands_xyz_file (str): Name of the ligand subgraph xyz file
        most_stable (bool): Wether to use the most stable occurence xyz of a ligand or the file specified by ligands_xyz_file

    Returns:
        tuple : (mol, coordination_atom_indices, stable_occ)
    """
    if most_stable:
        ligands_xyz_file = get_stable_occ(row.name)
    coordination_atom_indices = get_connection_ids(row, ligands_xyz_file)

    flattened_indices = [
        item for sublist in coordination_atom_indices for item in sublist
    ]

    # Load dict if it does not exist
    if "ligand_xyzs" not in locals():
        ligand_xyzs = data_handler.load_ligand_xyz()

    # Get the ligand xyz file from ligand xyz dict.
    xyz = ligand_xyzs[ligands_xyz_file]
    charge = int(row["charge"])
    mol = get_lig_mol_fixed_charge(xyz, charge, flattened_indices)
    if not mol:
        return None

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        _logger.info(f"Sanitation failed for {row.name} with : {e}")
        return None

    # We return the mol, coordinating atom idxs and the name of the TMC from which the ligand xyzfile was obtained.
    return (mol, coordination_atom_indices)


def get_tmcomplex_ligands():
    """Get dictionary with TMC label as key and ligand labels as values.

    Indicates the composition of each TMC.
    """
    if not os.path.exists(os.path.join(__location__, "tm_complex_ligands.pkl")):
        print("TM complex dict not existing. Creating it now.")
        complex_dict = defaultdict(list)
        df, df2 = data_handler.get_all_ligands_dfs()
        for row in tqdm.tqdm(df2.itertuples()):
            metal_dict = ast.literal_eval(row.parent_metal_occurrences)

            # for k,v in metal_dict.items():
            for k, labels in metal_dict.items():
                for label in labels:
                    complex_dict[label.split("-")[0]].append((row.Index, label))

        with open(os.path.join(__location__, "tm_complex_ligands.pkl"), "wb") as f:
            pickle.dump(complex_dict, f)
        print("Done with creating dict")

    # Get dict of compositions of each TMC
    with open(os.path.join(__location__, "tm_complex_ligands.pkl"), "rb") as f:
        tm_complex_dict = pickle.load(f)

    return tm_complex_dict


def get_tmc_smiles(args):
    # Create output directory
    output_dir = Path(".") / f"run_get_tm_{time.strftime('%Y-%m-%d_%H-%M')}"
    output_dir.mkdir(exist_ok=True)

    # Setup logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "printlog.txt", mode="w"),
            # logging.StreamHandler(),  # For debugging. Can be removed on remote
        ],
    )
    _logger.info("Current git hash: %s", get_git_revision_short_hash())

    tm_complex_dict = get_tmcomplex_ligands()

    # For debugging a small sample in parallell. Create sample dict
    if args.debug:
        sample = {}
        for i, (k, v) in enumerate(tm_complex_dict.items()):
            if i > 0:
                sample[k] = v
                if i == 50:
                    tm_complex_dict = sample
                    break

    # For debugging single structure non parallell
    # complex_label = "SAVREP"
    # m, com = process_complex((complex_label, tm_complex_dict[complex_label]))
    # print(m, com, "Done")
    # sys.exit(0)

    print(f"Processing: {len(tm_complex_dict)} complexes")
    start = time.time()
    mols = []
    # Process in parallell with timeout on each process.
    mols = process_paralell(
        process_complex,
        [(complex, ligands) for (complex, ligands) in tm_complex_dict.items()],
        num_workers=6,
        timeout=10,
    )

    # Get only the ones where we got a valid mol object out.
    valid_smiles = [comp[0] for comp in mols if (comp and comp[0])]
    res = [x for x in valid_smiles]

    df = pd.DataFrame(res)
    df.to_csv(output_dir / "output_tm_smiles.csv", index=False)

    end = time.time()
    _logger.info(f"Total TMC smiles: {len(df)}")
    _logger.info(f"Total time: {end - start}")


def parse_args(arg_list: list = None) -> argparse.Namespace:
    """Parse arguments from command line.

    Args:
        arg_list (list, optional): Automatically obtained from the command line if provided.
        If no arguments are given but default arguments are defined, the latter are used.

    Returns:
        argparse.Namespace: Dictionary-like class that contains the driver arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--function",
        "-f",
        choices=[
            "get_tm_smiles",
        ],
        default="get_tm_smiles",
        help="Which function to run.",
    )
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args(arg_list)


def main():
    FUNCTION_MAP = {
        "get_tm_smiles": get_tmc_smiles,
    }

    args = parse_args()
    _logger.info(f"Command line args: {args}")
    func = FUNCTION_MAP[args.function]

    func(args)

    sys.exit(0)

    return


if __name__ == "__main__":
    df1, df2 = data_handler.get_all_ligands_dfs()
    tmQMg_properties = data_handler.get_tmQMg_df()
    # Get the dict with tmc ligand composition
    ligand_xyzs = data_handler.load_ligand_xyz()
    main()

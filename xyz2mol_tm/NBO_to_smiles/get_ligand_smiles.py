"""Module containing functionality to get SMILES for ligands in tmQMg-L."""

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import data_handler
import mol_utils
import pandas as pd
from get_tmc_smiles import get_smiles_for_tmqmg_ligand
from mol_utils import get_git_revision_short_hash, process_paralell

ALLOWED_OXIDATION_STATES, TRANSITION_METALS = (
    mol_utils.ALLOWED_OXIDATION_STATES,
    mol_utils.TRANSITION_METALS,
)

from rdkit import Chem

_logger: logging.Logger = logging.getLogger("tmc_smiles")


__location__ = Path(__file__).absolute()
__root__ = Path(__file__).parent.parent.absolute()


def get_smiles_from_ligand_label(ligand):
    # Function for getting a smiles for a ligand in the tmQMg-L.

    ligand_data = defaultdict(dict)
    _logger.info(f"Processing: {ligand}")
    ligand_row = df2.loc[ligand]
    if ligand_row.empty:
        _logger.warning(f"Ligand {ligand} not found in data.")
        return None

    ligand_result = get_smiles_for_tmqmg_ligand(ligand_row, "", most_stable=True)
    if ligand_result:
        dict_entry = ligand
        mol, connection_ids = ligand_result
        ligand_data[dict_entry] = {
            "connection_atoms": connection_ids,
            "mol": mol,
            "smi": Chem.MolToSmiles(mol),
        }
    else:
        _logger.warning(f"Failed to generate mol object for {ligand}")
        dict_entry = ligand
        ligand_data[dict_entry] = None

    try:
        Chem.MolFromSmiles(ligand_data[dict_entry]["smi"])
    except Exception:
        _logger.info(f"Failed MolFromSmiles for: {ligand}")
        return None, None

    _logger.info(f"Successfully processed: {ligand}")

    return ligand_data


def get_ligand_smiles(args):
    output_dir = Path(".") / f"run_get_ligand_{time.strftime('%Y-%m-%d_%H-%M')}"
    output_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "printlog.txt", mode="w"),
            # logging.StreamHandler(),  # For debugging. Can be removed on remote
        ],
    )

    _logger.info("Current git hash: %s", get_git_revision_short_hash())

    # For debugging non parallell
    # lig = "ligand2-0"
    # get_smiles_from_ligand_label(lig)

    df, df2 = data_handler.get_all_ligands_dfs()
    names = df.index.tolist()
    names = names[0:10]

    start = time.time()
    mols = []

    mols = process_paralell(
        get_smiles_from_ligand_label, names, num_workers=6, timeout=160
    )

    combined_dict = {k: v for d in mols for k, v in d.items()}

    df = pd.DataFrame.from_dict(combined_dict, orient="index")
    df.to_csv(output_dir / "output_ligands_smiles.csv")

    end = time.time()
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
            "get_ligand_smiles",
        ],
        default="get_ligand_smiles",
        help="Which function to run",
    )

    parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        default=".",
        help="Output dir ",
    )
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args(arg_list)


def main():
    FUNCTION_MAP = {
        "get_ligand_smiles": get_ligand_smiles,
    }

    args = parse_args()
    print(args)
    func = FUNCTION_MAP[args.function]

    func(args)

    sys.exit(0)

    return


if __name__ == "__main__":
    ligand_xyzs = data_handler.load_ligand_xyz()
    df, df2 = data_handler.get_all_ligands_dfs()
    main()

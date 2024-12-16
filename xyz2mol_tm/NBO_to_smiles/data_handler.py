"Module handling the tmQMg datasets"

import ast
import itertools
import json
import os
from pathlib import Path
from typing import List

import pandas as pd

__current__ = Path(__file__).parent.absolute()

PATH_TO_TMQMG = __current__ / Path("../../tmQMg")
PATH_TO_TMQMG_L = __current__ / Path("../../tmQMg-L")

if not PATH_TO_TMQMG_L.exists() or not PATH_TO_TMQMG.exists():
    raise Exception(
        "tmQMg-L or tmQMg does not exist. Please download these repos and put them at the root level"
    )

ligands_xyz = PATH_TO_TMQMG_L / "xyz/ligands_xyzs.xyz"


def check_dict(string, tm_label):
    "Function to check if a certain TMC is in the ligand metal_bond_node_idx_groups dict"
    metal_dict = ast.literal_eval(string)

    labels = list(itertools.chain.from_iterable(metal_dict.values()))

    unique_labels = set([x.split("-")[0] for x in labels])

    if tm_label in unique_labels:
        return True
    else:
        return False


def load_csv(file, sep=",", **kwargs):
    df = pd.read_csv(file, sep=sep, **kwargs)
    return df


def get_tmQMg_df():
    tmQMg_properties = load_csv(
        PATH_TO_TMQMG / "tmQMg_properties_and_targets.csv", sep=","
    )

    return tmQMg_properties


def load_ligand_xyz():
    "Get the dict with tmQMg-L ligand xyz files"
    if not os.path.isfile(
        os.path.join(__current__, "data_files/ligands_dict_xyz.json")
    ):
        print("Dict with ligand xyz coordinates does not excist. Creating it now.")
        make_dict_xyz()

    # load ligand xyz dict
    with open(os.path.join(__current__, "data_files/ligands_dict_xyz.json"), "r") as f:
        xyzs = json.load(f)

    return xyzs


def make_dict_xyz():
    """Create dict from the long ligand xyz file from the tmQMg-L dataset.

    The keys in the dict will be the name of the TM for which the ligand
    xyz is found, with the corresponding xyz coordinate string as the
    values
    """
    # load xyzs
    xyzs = {}
    with open(str(ligands_xyz), "r") as fh:
        for xyz in fh.read().split("\n\n"):
            xyzs[xyz.split("\n")[1]] = xyz

    # Write to didct
    with open(os.path.join(__current__, "data_files/ligands_dict_xyz.json"), "w") as f:
        json.dump(xyzs, f)
    print("Succesfully created the dict of stable ligand complexes")


def get_all_ligands_dfs():
    "Load both of the csv files from the tmQMg-L dataset"
    df = load_csv(
        PATH_TO_TMQMG_L / "ligands_fingerprints.csv",
        sep=";",
    ).set_index("name")
    df2 = load_csv(PATH_TO_TMQMG_L / "ligands_misc_info.csv", sep=";").set_index("name")

    df2["charge"] = df["charge"]

    return df, df2


def get_monodentate():
    "Get df of neutral monodentates"

    df = load_csv(PATH_TO_TMQMG_L / "ligands_fingerprints.csv", sep=";")
    df2 = load_csv(PATH_TO_TMQMG_L / "ligands_misc_info.csv", sep=";")

    df2["charge"] = df["charge"]
    monodentate = df2[(df["n_metal_bound"] == 1) & (df["charge"] == 0)]
    return monodentate


def get_bidentate(df, df2, charge=0):
    "Get df of neutral mono and bidentates"

    df = load_csv(PATH_TO_TMQMG_L / "ligands_fingerprints.csv", sep=";")
    df2 = load_csv(PATH_TO_TMQMG_L / "ligands_misc_info.csv", sep=";")
    df2["charge"] = df["charge"]
    # mono_mask = (df["n_metal_bound"] == 1) & (df["n_dentic_bound"] == 1)
    bi_mask = (df["n_metal_bound"] == 2) & (df["n_dentic_bound"] == 2)
    charge_mask = df2["charge"] == charge
    monodentate = df2[(bi_mask) & charge_mask]
    return monodentate


def get_stable_occ(name):
    "Get string of stable occurence label for a ligand"

    ligands_stable_df = load_csv(PATH_TO_TMQMG_L / "stable.csv", sep=";")
    stable_oc = ligands_stable_df[ligands_stable_df["name"] == name][
        "stable_occurrence_name"
    ].item()
    return stable_oc


def get_connection_ids(row, ligand_xyz_file):
    "Extract the connection id of ligand as list where id can be accessed"
    # stable_oc = get_stable_occ(row["name"])
    res = row["metal_bond_node_idx_groups"]
    ocs = ast.literal_eval(res)
    id = ocs[ligand_xyz_file]
    return id


def load_file(file_path: str) -> List[str]:
    """Load the lines of a file."""
    with open(file_path, "r") as f:
        return f.readlines()

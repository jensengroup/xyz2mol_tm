"Module for getting data from the CSD using the python API"

import argparse
import sys
import time
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from ccdc.io import EntryReader
from pebble import ProcessExpired, ProcessPool
from rdkit import Chem

TRANSITION_METALS = [
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "La",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
]


def process_paralell(function, arguments, num_workers=6, timeout=30):
    res = []
    with ProcessPool(max_workers=num_workers) as pool:
        future = pool.map(
            function,
            [id for id in arguments],
            timeout=timeout,
        )
        iterator = future.result()

        i = 0
        while True:
            if i % 100 == 0:
                print("Finished {i} iterations")
            try:
                result = next(iterator)
                # print(result)
                res.append(result)
            except StopIteration:
                print("Stop iteration")
                break
            except TimeoutError:
                print("Timeout error")
                res.append(None)
            except ProcessExpired as error:
                print("%s. Exit code: %d" % (error, error.exitcode))
                res.append(None)
            except Exception as error:
                print(arguments[i])
                print("function raised %s" % error)
                res.append(None)
                # print(error.traceback)  # Python's traceback of remote process
            i += 1
    return res


class CSD_smiles:
    def __init__(self, entry):
        self.entry = entry  # CSD indentifier

    def get_smiles(self):
        try:
            self.smiles = self.entry.molecule.heaviest_component.smiles
            # Ned the formula for possibly filtering out the smiles
            self.formula = self.entry.molecule.heaviest_component.formula
        except RuntimeError as e:
            print(e)
            print(f"{self.entry.identifier} threw an exception when quering SMILES")
            self.smiles = "API_smiles_exception"

        if not self.smiles:
            print(f"{self.entry.identifier} has no SMILES entry")
            self.smiles = "API_smiles_missing"

        smiles = self.screen_query_result()
        if not self.smiles:
            print(f"{self.entry.identifier} Smiles failed the filter function")
            self.smiles = "SMILES_filter_fail"

        return smiles

    def screen_query_result(self):
        """Screen the resulting query for the correct smiles."""
        # Check for fragment

        # If there are multiple TM fragments pass.
        matches = [self.formula.count(x) for x in TRANSITION_METALS]
        if sum(matches) > 1:
            print("Too many TMs")
            return None

        # Get Heaviest fragment
        if self.smiles and "." in self.smiles:
            print("Getting smiles for heaviest fragment")
            self.smiles = self.get_tm_fragment()

        if not self.smiles:
            return self.smiles

        # Ensure that the selected smiles have a TM
        if not any(x in self.formula for x in TRANSITION_METALS):
            print("No TM")
            return None

        return self.smiles

    def get_tm_fragment(self):
        "Process fragmented string"
        heaviest = self.entry.molecule.heaviest_component.smiles
        if not heaviest:
            return None
        else:
            if any(x in heaviest for x in TRANSITION_METALS):
                return heaviest
        return None


class CSD_caller:
    def __init__(self, csd_id, properties=None):
        self.csd_id = csd_id

        self.csd_reader = EntryReader("CSD")
        if not self.csd_reader:
            raise ValueError(f"Id not found in database: {csd_id}")

        self.entry = self.csd_reader.entry(self.csd_id)
        self.smiles_getter = CSD_smiles(self.entry)

        if not properties:
            raise ValueError("No properties asked for")
        self.properties = properties

    def get_properties(self):
        # Use ordered dict to store properties so the order match the properties list
        results = OrderedDict()
        for prop in self.properties:
            if prop == "smiles":
                results[prop] = self.smiles_getter.get_smiles()
            elif prop == "charge":
                results[prop] = str(
                    self.entry.molecule.heaviest_component.formal_charge
                )
            elif prop == "formula":
                # The formula can contain a comma so we wrap in quotes
                results[prop] = f"{self.entry.molecule.heaviest_component.formula}"
            elif prop == "xyz":
                mol2_block = self.entry.molecule.heaviest_component.to_string()
                mol2 = Chem.MolFromMol2Block(mol2_block, sanitize=False)
                xyz_str = Chem.MolToXYZBlock(mol2)
                results["xyz_heaviest_fragment"] = xyz_str
            else:
                raise ValueError(f"Not valid property: {prop}")

        return results


def query_csd(input_args):
    csd_id, args = input_args

    # Setup caller
    caller = CSD_caller(csd_id, args.properties)

    # Get the requested properties
    result = caller.get_properties()

    print(result)

    # Write the properties to file
    with open(args.query_output_file, "a") as f:
        f.write(csd_id + "," + ",".join(list(result.values())) + "\n")

    return result


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
            "get_properties",
        ],
        default="get_properties",
        help="Which function to run",
    )
    parser.add_argument(
        "--query_output_file", type=Path, default="query_csd_output.csv"
    )
    parser.add_argument(
        "--properties",
        "-p",
        type=str,
        nargs="+",
        default=["smiles", "formula"],
        help="Which properties to get",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=6, help="Number of cores to parallelize over"
    )

    return parser.parse_args(arg_list)


def main(args):
    func = FUNCTION_MAP[args.function]

    df = pd.read_csv("../../SMILES_csvs/tmqmg_csd_api_data.csv")
    ids = df["IDs"].values
    if args.debug:
        ids = ids[0:10]

    # ids = [("ABIZIW", args)]
    # ids.extend([("ABIZIW", args)]
    # res = func(("ROSJUH", args))
    # sys.exit()

    start = time.time()
    print(ids[0:5], len(ids))

    if args.query_output_file.exists():
        raise ValueError("Output file already exists")
    with open(args.query_output_file, "a") as f:
        f.write(",".join(args.properties) + "\n")

    input_args = [(id, args) for id in ids]

    process_paralell(func, input_args, num_workers=args.workers)

    end = time.time()
    print(f"{end - start:.2f} seconds")

    return


if __name__ == "__main__":
    FUNCTION_MAP = {
        "get_properties": query_csd,
    }

    # For debug
    # tmqmg_id = "IVUZIL"
    # res = csd_wrapper(tmqmg_id)

    args = parse_args()
    print(args)

    main(args)
    sys.exit(0)

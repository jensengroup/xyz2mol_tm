import os
import re
import sys
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.warning")


module_path = os.path.abspath("../")
if module_path not in sys.path:
    sys.path.append(module_path)

__location__ = Path(__file__).absolute()
__root__ = Path(__file__).parent.parent.absolute()

from xyz2mol_tm import xyz2mol_local
from xyz2mol_tm.NBO_to_smiles.mol_utils import TRANSITION_METALS, xyz_string_decompose

tms = TRANSITION_METALS
tm = f"[{','.join(tms)}]"

tm_ms = [Chem.MolFromSmiles(f"[{x}]") for x in tms]
tm_nums = [x.GetAtoms()[0].GetAtomicNum() for x in tm_ms]


def contains_substring(string):
    substrings = tms

    for substring in substrings:
        if substring in string:
            return substring
    return False


def filter_function(smi):
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if not m:
        return False

    allowed_elements = (
        [i for i in range(5, 11)]
        + [i for i in range(13, 19)]
        + [i for i in range(31, 37)]
        + [i for i in range(49, 55)]
        + [i for i in range(81, 87)]
        + [i for i in range(113, 119)]
        + [1]
        + tm_nums
    )

    if m.GetNumAtoms() > 100:
        return False

    matches = m.GetSubstructMatches(Chem.MolFromSmarts(tm))
    if (len(matches) > 1) or (len(matches) == 0):
        print(matches)
        return False

    elements = [a.GetAtomicNum() for a in m.GetAtoms()]
    [a.GetSymbol() for a in m.GetAtoms()]
    if any(x not in allowed_elements for x in elements):
        return False
    # if not any(x in elements_sym for x in ['Cu', 'Cr']):
    #     return False

    return True


def is_valid_formula(molecular_formula):
    # Define allowed elements and transition metals
    transition_metals = {
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
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
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Nh",
        "Fl",
        "Mc",
        "Lv",
        "Ts",
    }

    allowed_elements = {
        "H",
        "B",
        "Al",
        "Ga",
        "In",
        "Tl",
        "C",
        "Si",
        "Ge",
        "Sn",
        "Pb",
        "N",
        "P",
        "As",
        "Sb",
        "Bi",
        "O",
        "S",
        "Se",
        "Te",
        "Po",
        "F",
        "Cl",
        "Br",
        "I",
        "At",
    }

    # Regular expression to match element symbols and counts
    element_regex = re.compile(r"([A-Z][a-z]*)(\d*)")

    # Split the formula into parts and charge
    parts = molecular_formula.split()

    # Check if the last part is a valid charge
    if parts[-1] in ["+", "-"] or re.match(r"^[+-]?\d+[+-]?$", parts[-1]):
        parts.pop()  # Remove the charge from the parts

    molecular_formula = " ".join(
        parts
    )  # Rebuild the molecular formula without the charge

    # Variables to keep track of transition metals and heavy atoms
    transition_metal_count = 0
    heavy_atom_count = 0

    # Parse the molecular formula
    matches = element_regex.findall(molecular_formula)

    for element, count_str in matches:
        count = (
            int(count_str) if count_str else 1
        )  # If no count is provided, assume it's 1

        if element in transition_metals:
            transition_metal_count += count  # Add the number of transition metals
        elif element in allowed_elements:
            if element != "H":  # Count heavy atoms (all except hydrogen)
                heavy_atom_count += count
        else:
            # If we find an element that is neither a transition metal nor in groups 13-17, return False
            return False

    # Check the conditions
    if transition_metal_count != 1:  # We need exactly one transition metal
        return False
    if heavy_atom_count > 100:  # No more than 100 heavy atoms
        return False

    return True


class FormulaParser:
    def __init__(self, formula):
        self.formula = formula
        # self.smiles = smiles

        self.tm = f"[{','.join(tms)}]"

    def get_hydrogen_in_formula(self):
        pattern = r"H\d+"

        match = re.search(pattern, self.formula)
        if match:
            return match.group(0)
        else:
            return 0

    @staticmethod
    def get_element_in_formula(formula, pattern="H\d*"):
        match = re.search(pattern, formula)
        if match:
            return match.group(0)
        else:
            return None

    def get_charge_in_formula(self):
        if not self.formula:
            return None

        pattern = r"(?<!\w)([+-]?\d+[+-]?)$"
        match = re.search(pattern, self.formula)
        if match:
            return match.group(0)
        else:
            return 0


def parse_formula(formula, task=None):
    if formula != formula:
        return None

    parser = FormulaParser(formula)

    if task == "hydrogen":
        result = parser.get_hydrogen_in_formula()
    elif task == "charge":
        result = parser.get_charge_in_formula()
    else:
        raise ("Invalid option")

    return result


def get_formula_from_smiles(smiles):
    m = Chem.MolFromSmiles(smiles)
    form = rdMolDescriptors.CalcMolFormula(m)

    return form

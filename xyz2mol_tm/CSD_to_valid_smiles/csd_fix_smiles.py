# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %%
import argparse
import re
import sys

# %%
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Draw, rdchem, rdFMCS, rdMolDescriptors
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.MolStandardize import rdMolStandardize

bond2charge = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3}

atom_list = [
    "h",
    "he",
    "li",
    "be",
    "b",
    "c",
    "n",
    "o",
    "f",
    "ne",
    "na",
    "mg",
    "al",
    "si",
    "p",
    "s",
    "cl",
    "ar",
    "k",
    "ca",
    "sc",
    "ti",
    "v",
    "cr",
    "mn",
    "fe",
    "co",
    "ni",
    "cu",
    "zn",
    "ga",
    "ge",
    "as",
    "se",
    "br",
    "kr",
    "rb",
    "sr",
    "y",
    "zr",
    "nb",
    "mo",
    "tc",
    "ru",
    "rh",
    "pd",
    "ag",
    "cd",
    "in",
    "sn",
    "sb",
    "te",
    "i",
    "xe",
    "cs",
    "ba",
    "la",
    "ce",
    "pr",
    "nd",
    "pm",
    "sm",
    "eu",
    "gd",
    "tb",
    "dy",
    "ho",
    "er",
    "tm",
    "yb",
    "lu",
    "hf",
    "ta",
    "w",
    "re",
    "os",
    "ir",
    "pt",
    "au",
    "hg",
    "tl",
    "pb",
    "bi",
    "po",
    "at",
    "rn",
    "fr",
    "ra",
    "ac",
    "th",
    "pa",
    "u",
    "np",
    "pu",
]

TMs = {
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    57,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
}

TM_nums = "[#21,#22,#23,#24,#25,#26,#27,#28,#29,#30,#39,#40,#41,#42,#43,#44,#45,#46,#47,#48,#57,#72,#73,#74,#75,#76,#77,#78,#79,#80]"


def extract_bracketed_strings(s):
    pattern = r"\[([^\]]+)\]"
    return re.findall(pattern, s)


def get_smiles(sm):
    tm_comp = None
    counter_ions = []
    for s in sm.split("."):
        atoms = extract_bracketed_strings(s)
        # print('get_smiles',s,atoms)

        if atoms:
            for atom in atoms:
                atom = atom.split("+")[0]
                atom = atom.split("-")[0]
                if "Hg" not in atom and "Hf" not in atom:
                    atom = atom.split("H")[0]
                atom = atom.lower()
                if atom in atom_list:
                    atom_num = atom_list.index(atom) + 1
                    # print(s,atom,atom_num)
                    if atom_num in TMs:
                        tm_comp = s
            if not tm_comp:
                counter_ions.append(s)
        else:
            counter_ions.append(s)

    return tm_comp, counter_ions


def rem_chirality(m):
    for atom in m.GetAtoms():
        if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)

    return m


def fix_NtripleTM(m):
    hit = m.GetSubstructMatches(Chem.MolFromSmarts("*N#" + TM_nums))

    for b in hit:
        _, a, tm = b
        m.GetAtomWithIdx(a).SetFormalCharge(1)
        tm_atom = m.GetAtomWithIdx(tm)
        tm_atom.SetFormalCharge(tm_atom.GetFormalCharge() - 1)

    return m


def fix_OtripleTM(m):
    hit = m.GetSubstructMatches(Chem.MolFromSmarts("O#" + TM_nums))

    for b in hit:
        a, tm = b
        m.GetAtomWithIdx(a).SetFormalCharge(1)
        tm_atom = m.GetAtomWithIdx(tm)
        tm_atom.SetFormalCharge(tm_atom.GetFormalCharge() - 1)

    return m


def fix_NO(m):
    hit = m.GetSubstructMatches(Chem.MolFromSmarts("nO" + TM_nums))

    for c in hit:
        a, b, tm = c
        mw = Chem.RWMol(m)
        mw.RemoveBond(b, tm)
        mw.AddBond(b, tm, Chem.BondType.DATIVE)
        m = mw.GetMol()

        m.GetAtomWithIdx(a).SetFormalCharge(1)
        m.GetAtomWithIdx(b).SetFormalCharge(-1)
        tm_atom = m.GetAtomWithIdx(tm)
        tm_atom.SetFormalCharge(tm_atom.GetFormalCharge() - 1)

    hit = m.GetSubstructMatches(Chem.MolFromSmarts("nO"))

    for c in hit:
        a, b = c

        m.GetAtomWithIdx(a).SetFormalCharge(1)
        m.GetAtomWithIdx(b).SetFormalCharge(-1)

    return m


def fix_aromatic_tertn(m):
    tertns = [x[0] for x in m.GetSubstructMatches(Chem.MolFromSmarts("[nD3+0;!$(n" + TM_nums + ")]"))]
    arings = list(m.GetSubstructMatches(Chem.MolFromSmarts("a1aaaa1"))) + list(m.GetSubstructMatches(Chem.MolFromSmarts("a1aaaaa1")))
    tm = m.GetSubstructMatches(Chem.MolFromSmarts(TM_nums))[0][0]

    for ar in arings:
        atoms = list(set(ar).intersection(set(tertns)))
        if not atoms:
            continue
        atom = atoms[0]
        m.GetAtomWithIdx(atom).SetFormalCharge(1)
        tm_atom = m.GetAtomWithIdx(tm)
        tm_atom.SetFormalCharge(tm_atom.GetFormalCharge() - 1)

    return m


def covalent2haptic(mol, sanitize=True, fix_charge=True, fix_counter_charge=True):
    # print('in covalent2haptic')
    """Written by Julius Seumer Converts covalent bonds to transition metal
    from neighbouring atoms in a molecule to haptic bonds.

    Args:
            mol (rdkit.Chem.rdchem.Mol): The molecule to convert.
            sanitize (bool, optional): Whether to sanitize the molecule after conversion. Defaults to True.
            fix_charge (bool, optional): Whether to fix the charge of the haptic ligand when number of atoms
                                                                      that are part of haptic bond is uneven. Defaults to True.
            fix_counter_charge (bool, optional): Whether to fix the counter charge on the transition metal.
                                                                                      Defaults to True.

    Returns:
            rdkit.Chem.rdchem.Mol: The converted molecule with haptic bonds.
    """

    TRANSITION_METALS = "[Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,Lu,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg]"

    def _dfs(visited, graph, node):
        if node not in visited:
            visited.add(node)
            for neighbour in graph[node]:
                _dfs(visited, graph, neighbour)

    tm_ids = mol.GetSubstructMatches(Chem.MolFromSmarts(TRANSITION_METALS))
    for tm_idx in tm_ids:
        tm = mol.GetAtomWithIdx(tm_idx[0])
        # get all bonds to metal
        bonds = set(tm.GetBonds())
        other_atom_ids = [b.GetOtherAtomIdx(tm_idx[0]) for b in bonds]
        # cluster bonds
        adj = Chem.GetAdjacencyMatrix(mol)
        for i in [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in other_atom_ids]:
            adj[:, i] = 0
            adj[i, :] = 0
        graph = {i: np.nonzero(row)[0].tolist() for i, row in enumerate(adj)}
        checked = set()
        clusters = []
        for other_idx in other_atom_ids:
            if other_idx in checked:
                continue
            visited = set()
            _dfs(visited, graph, other_idx)
            clusters.append(list(visited))
            checked.update(visited)
        # if a cluster is longer then 1, then it is a haptic bond
        for cluster in clusters:
            if len(cluster) == 2 and mol.GetBondBetweenAtoms(cluster[0], cluster[1]).GetBondType() == Chem.BondType.SINGLE:
                continue
            if len(cluster) > 1:
                for c_idx in cluster:
                    # convert all covalent bonds in cluster to dative
                    # needs to remove/add bond to get the right direction
                    emol = Chem.EditableMol(mol)
                    emol.RemoveBond(c_idx, tm.GetIdx())
                    emol.AddBond(c_idx, tm.GetIdx(), Chem.BondType.DATIVE)
                    mol = emol.GetMol()
                # add negative charge if cluster has uneven number of atoms
                if fix_charge:
                    if len(cluster) % 2 == 1:
                        # look if there is already a negative charge and skip cluster
                        if sum([a.GetFormalCharge() for a in mol.GetAtoms() if a.GetIdx() in cluster]) < 0:
                            continue
                        # find atom with least neighbors and set negative charge there
                        degrees = [a.GetDegree() for a in mol.GetAtoms() if a.GetIdx() in cluster]
                        min_degree_idx = cluster[np.argmin(degrees)]
                        atom = mol.GetAtomWithIdx(min_degree_idx)
                        atom.SetFormalCharge(-1)
                        if atom.GetDegree() <= 3:
                            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
                        if fix_counter_charge:
                            # increment formal charge on TM in new mol
                            tm = mol.GetAtomWithIdx(tm_idx[0])
                            tm.SetFormalCharge(tm.GetFormalCharge() + 1)
    # if sanitize:
    #    Chem.SanitizeMol(mol)
    # haptic_mol = Chem.DativeBondsToHaptic(mol)
    return mol


def fix_smiles(smi):
    pt = Chem.GetPeriodicTable()
    m = Chem.MolFromSmiles(smi, sanitize=False)
    tm_bonds = m.GetSubstructMatches(Chem.MolFromSmarts("*~" + TM_nums))
    # m = fix_carbenes(m)
    # m = fix_Ntriplebond(m)
    # m = fix_CO(m)
    # m = fix_B(m)
    # m = fix_NO2(m)
    m = fix_NtripleTM(m)
    m = fix_OtripleTM(m)
    m = fix_NO(m)
    m = fix_aromatic_tertn(m)
    m = covalent2haptic(m)

    tm_bonds = m.GetSubstructMatches(Chem.MolFromSmarts("[*]-,=,#" + TM_nums))
    bond_types = [m.GetBondBetweenAtoms(i, j).GetBondType() for i, j in tm_bonds]
    bond_types = [str(x).split(".")[-1] for x in bond_types]

    mw = Chem.RWMol(m)
    for i, tm in tm_bonds:
        mw.RemoveBond(i, tm)
    m = mw.GetMol()

    problems = Chem.DetectChemistryProblems(m)
    for error in problems:
        if error.GetType() != "AtomValenceException":
            continue
        fix_atoms = [error.GetAtomIdx()]
        for atom in fix_atoms:
            if m.GetAtomWithIdx(atom).GetAtomicNum() == 5:
                d_charge = -1
            else:
                d_charge = 1
            m.GetAtomWithIdx(atom).SetFormalCharge(d_charge)
            tm_atom = m.GetAtomWithIdx(tm)
            tm_atom.SetFormalCharge(tm_atom.GetFormalCharge() - d_charge)

    l_atoms = set([x[0] for x in tm_bonds])
    for i in range(6):
        # while problems:
        problems = Chem.DetectChemistryProblems(m)
        for error in problems:
            if error.GetType() != "KekulizeException":
                continue
            fix_atoms = l_atoms.intersection(set(error.GetAtomIndices()))
            for atom in fix_atoms:
                m.GetAtomWithIdx(atom).SetFormalCharge(-1)
                tm_atom = m.GetAtomWithIdx(tm)
                tm_atom.SetFormalCharge(tm_atom.GetFormalCharge() + 1)

    Chem.SanitizeMol(m)

    mw = Chem.RWMol(m)
    for i, tm in tm_bonds:
        mw.AddBond(i, tm, Chem.BondType.DATIVE)
    m = mw.GetMol()

    for (i, tm), bt in zip(tm_bonds, bond_types):
        nbr = m.GetAtomWithIdx(i)
        if pt.GetDefaultValence(nbr.GetAtomicNum()) - nbr.GetExplicitValence() + nbr.GetFormalCharge() <= 0:
            continue  # dative bond

        charge = bond2charge[bt]
        nbr.SetFormalCharge(nbr.GetFormalCharge() - charge)
        tm_atom = m.GetAtomWithIdx(tm)
        tm_atom.SetFormalCharge(tm_atom.GetFormalCharge() + charge)

    Chem.SanitizeMol(m)

    return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(m)))


def fix_carbene(m):
    Chem.Kekulize(m, clearAromaticFlags=True)
    hit = m.GetSubstructMatches(Chem.MolFromSmarts("[N+]=[C-]->" + TM_nums))

    last_b = -1
    for c in hit:
        a, b, tm = c
        if b == last_b:
            continue
        mw = Chem.RWMol(m)
        mw.RemoveBond(a, b)
        mw.AddBond(a, b, Chem.BondType.SINGLE)
        mw.GetAtomWithIdx(a).SetFormalCharge(+0)
        mw.GetAtomWithIdx(b).SetFormalCharge(+0)
        mw.GetAtomWithIdx(b).SetNumRadicalElectrons(2)
        m = mw.GetMol()
        last_b = b

    hit = m.GetSubstructMatches(Chem.MolFromSmarts("N[CX3-,CX3-2]->" + TM_nums))

    last_b = -1
    for c in hit:
        a, b, tm = c
        if b == last_b:
            continue

        chg = m.GetAtomWithIdx(b).GetFormalCharge()
        m.GetAtomWithIdx(b).SetFormalCharge(+0)
        m.GetAtomWithIdx(b).SetNumRadicalElectrons(2)
        tm_atom = m.GetAtomWithIdx(tm)
        tm_atom.SetFormalCharge(tm_atom.GetFormalCharge() + chg)

        last_b = b

    Chem.SanitizeMol(m)

    return m


def fix_CO(m):
    hit = m.GetSubstructMatches(Chem.MolFromSmarts("O=[CX2H0+0]->" + TM_nums))

    for c in hit:
        a, b, tm = c
        mw = Chem.RWMol(m)
        mw.RemoveBond(a, b)
        mw.AddBond(a, b, Chem.BondType.TRIPLE)
        mw.GetAtomWithIdx(a).SetFormalCharge(1)
        mw.GetAtomWithIdx(b).SetFormalCharge(-1)
        mw.GetAtomWithIdx(b).SetNumRadicalElectrons(0)
        m = mw.GetMol()

    Chem.SanitizeMol(m)

    return m


def fix_nHTM(m):
    hits = m.GetSubstructMatches((Chem.MolFromSmarts("[nH]->" + TM_nums)))

    for c in hits:
        a, tm = c
        m.GetAtomWithIdx(a).SetFormalCharge(-1)
        m.GetAtomWithIdx(a).SetNumExplicitHs(0)
        tm_atom = m.GetAtomWithIdx(tm)
        tm_atom.SetFormalCharge(tm_atom.GetFormalCharge() + 1)

    Chem.SanitizeMol(m)

    return m


# %%
def main(args):
    # Get csv
    df = pd.read_csv(args.csd_smiles)
    df["smiles_csd_api"].fillna(value="", inplace=True)

    debug = False
    if debug:
        df = df[:100]

    results = []
    for i, (ID, s, formula) in enumerate(zip(df["IDs"], df["smiles_csd_api"], df["formula_heaviest_fragment"])):
        tm_comp, counter_ions = get_smiles(s)
        if not tm_comp:
            results.append((ID, s, "fail", "fail", formula))
            continue
        try:
            new_s = fix_smiles(tm_comp)
            m = Chem.MolFromSmiles(new_s)
            m = fix_nHTM(m)
            m = fix_carbene(m)
            new_s = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(m)))
            results.append((ID, s, tm_comp, new_s, formula))
        except Exception:
            results.append((ID, s, tm_comp, "fail", formula))

    df_results = pd.DataFrame(
        results,
        columns=["IDs", "CSD smiles", "TM CSD smiles", "new TM smiles", "formula"],
    )

    (
        df_results[(df_results["CSD smiles"] == "fail")].shape,
        df_results[(df_results["TM CSD smiles"] != "fail")].shape,
    )

    (
        df_results[(df_results["TM CSD smiles"] != "fail") & (df_results["new TM smiles"] == "fail")].shape,
        df_results[(df_results["new TM smiles"] != "fail")].shape,
    )

    unusual_charge = []
    for s in df_results["new TM smiles"]:
        if s == "fail":
            unusual_charge.append("N/A")
            continue
        sio = sys.stderr = StringIO()
        m = Chem.MolFromSmiles(s)
        if "Unusual charge" in sio.getvalue():
            unusual_charge.append("yes")
        else:
            unusual_charge.append("no")

    df_results["unusual charge"] = unusual_charge

    df_results[(df_results["unusual charge"] == "yes")].shape

    ion_pairs = []
    for s in df_results["new TM smiles"]:
        if s == "fail":
            ion_pairs.append(-1)
            continue
        m = Chem.MolFromSmiles(s)
        ion_pairs.append(
            len(m.GetSubstructMatches(Chem.MolFromSmarts("[*-]~[*-]"))) + len(m.GetSubstructMatches(Chem.MolFromSmarts("[*+]~[*+]")))
        )

    df_results["ion pairs"] = ion_pairs

    df_results.to_csv(
        args.csd_smiles.parent / "output_df.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script takes a dataframe with CSD SMILES as input and tries to make them RDKit parseable"
    )
    parser.add_argument("--csd_smiles", type=Path, help="The input dataframe", required=True)

    # Parse arguments
    args = parser.parse_args()
    main(args)

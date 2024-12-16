import logging
import subprocess

import numpy as np
import py3Dmol
from pebble import ProcessExpired, ProcessPool
from rdkit import Chem
from rdkit.Geometry import Point3D

from xyz2mol_tm.NBO_to_smiles.data_handler import PATH_TO_TMQMG

_logger: logging.Logger = logging.getLogger("tmc_smiles")

# fmt: off
global __ATOM_LIST__
__ATOM_LIST__ = ["h","he","li","be","b","c","n","o","f","ne","na","mg","al",
                 "si","p","s","cl","ar","k","ca","sc","ti","v","cr","mn","fe",
                 "co","ni","cu","zn","ga","ge","as","se","br","kr","rb","sr",
                 "y","zr","nb","mo","tc","ru","rh","pd","ag","cd","in","sn","sb",
                 "te","i","xe","cs","ba","la","ce","pr","nd","pm","sm","eu","gd",
                 "tb","dy","ho","er","tm","yb","lu","hf","ta","w","re","os","ir",
                 "pt","au","hg","tl","pb","bi","po","at","rn","fr","ra","ac","th",
                 "pa","u","np","pu",
]


TRANSITION_METALS = ["Sc","Ti","V","Cr","Mn","Fe","Co","La","Ni","Cu","Zn",
                     "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","Lu",
                     "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
]

TRANSITION_METALS_NUM = [21,22,23,24,25,26,27,57,28,29,30,39,40,41,
                         42,43,44,45,46,47,48,71,72,73,74,75,76,77,78,79,80,
]
# fmt: on


ALLOWED_OXIDATION_STATES = {
    "Sc": [3],
    "Ti": [3, 4],
    "V": [2, 3, 4, 5],
    "Cr": [2, 3, 4, 6],
    "Mn": [2, 3, 4, 6, 7],
    "Fe": [2, 3],
    "Co": [2, 3],
    "Ni": [2],
    "Cu": [1, 2],
    "Zn": [2],
    "Y": [3],
    "Zr": [4],
    "Nb": [3, 4, 5],
    "Mo": [2, 3, 4, 5, 6],
    "Tc": [2, 3, 4, 5, 6, 7],
    "Ru": [2, 3, 4, 5, 6, 7, 8],
    "Rh": [1, 3],
    "Pd": [2, 4],
    "Ag": [1],
    "Cd": [2],
    "La": [3],
    "Hf": [4],
    "Ta": [3, 4, 5],
    "W": [2, 3, 4, 5, 6],
    "Re": [2, 3, 4, 5, 6, 7],
    "Os": [3, 4, 5, 6, 7, 8],
    "Ir": [1, 3],
    "Pt": [2, 4],
    "Au": [1, 3],
    "Hg": [1, 2],
}


def draw_mol(
    mol,
    width=800,
    height=800,
    Hs=False,
    confId=-1,
    multipleConfs=False,
    atomlabel=False,
    hit_ats=None,
    gen_struct=None,
    trajectory=False,
):
    p = py3Dmol.view(width=width, height=height)

    if isinstance(mol, str):
        if "\n" in mol:
            p.addModel(mol, "xyz")
        else:
            xyz_f = open(mol)
            line = xyz_f.read()
            xyz_f.close()
            p.addModel(line, "xyz")
    else:
        if multipleConfs:
            for conf in mol.GetConformers():
                mb = Chem.MolToMolBlock(mol, confId=conf.GetId())
                p.addModel(mb, "sdf")
        else:
            mb = Chem.MolToMolBlock(mol, confId=confId)
            p.addModel(mb, "sdf")

    p.setStyle({"stick": {"radius": 0.17}, "sphere": {"radius": 0.4}})
    p.setStyle({"elem": "H"}, {"stick": {"radius": 0.17}, "sphere": {"radius": 0.28}})
    if atomlabel:
        p.addPropertyLabels("index")  # ,{'elem':'H'}
    p.setClickable(
        {},
        True,
        """function(atom,viewer,event,container) {
        if(!atom.label) {
            atom.label = viewer.addLabel(atom.index,{position: atom, backgroundColor: ‘white’, fontColor:‘black’});
        }}""",
    )

    p.zoomTo()
    p.update()


def read_file(file_name, num_mols):
    """Read smiles from file and return list of mols."""
    mols = []
    with open(file_name, "r") as file:
        for i, smiles in enumerate(file):
            mols.append(smiles)
            if i == num_mols:
                break
    return mols


def view_opt_structure(structs, idx):
    coordinates = structs[idx]
    xyz = write_xyz(coordinates)
    draw_mol(xyz, width=800, height=800)


def write_xyz_a_c(atoms, coords):
    """Write xyz str."""
    natoms = len(atoms)
    xyz = f"{natoms} \n# \n"
    for atomtype, coord in zip(atoms, coords):
        xyz += f"{atomtype}  {' '.join(list(map(str, coord)))} \n"
    xyz += "\n"
    return xyz


def write_xyz(elem):
    """Write xyz str."""
    atoms, coords = elem[0]
    ir = elem[2]
    homo_lumo = elem[1]

    natoms = len(atoms)
    xyz = f"{natoms} \n# Ir-charge : {ir:.3}, homo_lumo : {homo_lumo:.3}\n"
    for atomtype, coord in zip(atoms, coords):
        xyz += f"{atomtype}  {' '.join(list(map(str, coord)))} \n"
    xyz += "\n"
    return xyz


def comp(list1, list2):
    true_count = 0
    for val in list1:
        if val in list2:
            true_count += 1
    return true_count


def get_git_revision_short_hash() -> str:
    """Get the git hash of current commit if git repo."""
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def get_smiles_atomidx_mapping(mol):
    "Get the smiles and mapped atom numbering for mol object"
    mol = Chem.RemoveHs(mol)
    smi = Chem.MolToSmiles(mol)
    order = eval(mol.GetProp("_smilesAtomOutputOrder"))
    mapped_ids = {}
    for i in range(0, len(mol.GetAtoms())):
        mapped_id = np.where(np.array(order) == i)[0][0]
        mapped_ids[i] = mapped_id
    return smi, mapped_ids


def attach_ligands(combined_mol, ligand_data):
    emol = Chem.RWMol(combined_mol)
    emol.BeginBatchEdit()
    atom_ids = Chem.GetMolFrags(combined_mol)
    for i, (ligand_subgraph, data_dict) in enumerate(ligand_data.items()):
        for coordinating_atom_list in data_dict["connection_atoms"]:
            for connect_id in coordinating_atom_list:
                # add dative bond to metal.
                connection_atom_id = atom_ids[i + 1][connect_id]
                emol.AddBond(connection_atom_id, 0, Chem.BondType.DATIVE)

    # commit changes made and get mol
    emol.CommitBatchEdit()
    mol = emol.GetMol()
    return mol


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
            i += 1
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
    return res


def get_transition_metal_atom(complex):
    """Extract the transition metal atom from the complex XYZ file."""
    path = PATH_TO_TMQMG / f"tmQMg_xyzfiles/{complex}.xyz"
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines[2:]:
        atom = line.split()[0]
        if atom in TRANSITION_METALS_NUM:
            return atom, len(lines) - 2  # Return atom symbol and atom count
    return None, 0


def combine_molecules(metal, ligands):
    """Combine metal and ligand molecules into a single complex molecule."""
    combined_mol = metal
    for ligand, res in ligands.items():
        try:
            combined_mol = Chem.CombineMols(combined_mol, res["mol"])
        except Exception as e:
            print(e)
            _logger.error(f"Failed to combine mol for complex {complex}")
            return None
    return combined_mol


def set_metal_coordinates(mol, metal_idx, complex):
    """Set the 3D coordinates of the metal atom from the XYZ file."""
    path = PATH_TO_TMQMG / f"tmQMg_xyzfiles/{complex}.xyz"
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines[2:]:
        atom_data = line.split()
        if atom_data[0] in TRANSITION_METALS_NUM:
            x, y, z = map(float, atom_data[1:4])
            conf = mol.GetConformer()
            conf.SetAtomPosition(metal_idx, Point3D(x, y, z))
            break


def xyz_string_decompose(xyz_string):
    atoms = []
    xyz_coordinates = []
    atomic_symbols = []

    for line_number, line in enumerate(xyz_string.split("\n")):
        if line_number == 0:
            int(line)
        elif line_number == 1:
            if "charge=" in line:
                int(line.split("=")[1])
        else:
            if not line:
                continue
            atomic_symbol, x, y, z = line.split()
            atomic_symbols.append(atomic_symbol)
            xyz_coordinates.append([float(x), float(y), float(z)])
    atoms = [int_atom(atom) for atom in atomic_symbols]

    return atoms, xyz_coordinates


def int_atom(atom):
    """Convert str atom to integer atom."""
    global __ATOM_LIST__
    # print(atom)
    atom = atom.lower()
    return __ATOM_LIST__.index(atom) + 1


def strip_stereo(smi):
    if smi != smi:
        return smi

    m = Chem.MolFromSmiles(smi)
    Chem.RemoveStereochemistry(m)

    stripped = Chem.MolToSmiles(m)

    return stripped


if __name__ == "__main__":
    pass

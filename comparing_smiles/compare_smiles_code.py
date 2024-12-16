import numpy as np
from pebble import ProcessExpired, ProcessPool
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable, RegistrationHash, rdchem, rdEHTTools, rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize

params = Chem.MolStandardize.rdMolStandardize.MetalDisconnectorOptions()
params.splitAromaticC = True
params.splitGrignards = True
params.adjustCharges = False
MetalNon_Hg = "[#3,#11,#12,#19,#13,#21,#22,#23,#24,#25,#26,#27,#28,#29,#30,#39,#40,#41,#42,#43,#44,#45,#46,#47,#48,#57,#72,#73,#74,#75,#76,#77,#78,#79,#80]~[#1,B,#6,#14,#15,#33,#51,#16,#34,#52,Cl,Br,I,#85]"
mdis = rdMolStandardize.MetalDisconnector(params)
mdis.SetMetalNon(Chem.MolFromSmarts(MetalNon_Hg))


def count(lst):
    return sum(bool(x) for x in lst)


def check_smiles(smi):
    ## String flags that are present in the dataset ##

    # If CSD api did not return a SMILES
    if smi == "API_smiles_missing":
        return False
    # If a CSD identifier was no longer present in the database
    if smi == "not_in_database":
        return False

    # If the CSD smiles fix method failed
    if smi == "fail":
        return False

    # Check for nan value
    if smi != smi:
        return False
    # Check for None value
    if not smi:
        return False

    return True


def get_agreement_between_multiple_smiles(s1, s2, s3):
    a_h = Chem.CanonSmiles(s1)
    b_h = Chem.CanonSmiles(s2)
    c_h = Chem.CanonSmiles(s3)
    if a_h == b_h == c_h:
        return True
    else:
        return False


def get_agreement_between_smiles(s1, s2):
    a_h = Chem.CanonSmiles(s1)
    b_h = Chem.CanonSmiles(s2)
    if a_h == b_h:
        return True
    else:
        return False


# Disconnect ligands and compare the resonance sets
def check_smiles_fragments_resonance(smile_set):
    sm1, sm2 = smile_set
    if Chem.CanonSmiles(sm1) == Chem.CanonSmiles(sm2):
        return True

    try:
        m1_mol = Chem.MolFromSmiles(sm1)
        m1 = mdis.Disconnect(m1_mol)
        m2_mol = Chem.MolFromSmiles(sm2)
        m2 = mdis.Disconnect(m2_mol)

        if len(Chem.GetMolFrags(m2, asMols=True)) != len(
            Chem.GetMolFrags(m1, asMols=True)
        ):
            return False

        # For the disconnector we need to check this first
        if Chem.MolToSmiles(m1) == Chem.MolToSmiles(m2):
            return True

        sm1_d = []
        sm2_d = []
        for f1 in Chem.GetMolFrags(m1, asMols=True):
            f1 = Chem.MolToSmiles(f1)
            sm1_d += [f1]

        for f2 in Chem.GetMolFrags(m2, asMols=True):
            f2 = Chem.MolFromSmiles(Chem.MolToSmiles(f2))
            sm2_d += [
                Chem.CanonSmiles(Chem.MolToSmiles(x))
                for x in Chem.ResonanceMolSupplier(
                    f2, flags=Chem.ALLOW_CHARGE_SEPARATION
                )
            ]

            # print(f2_sm, sm1_d)
            # if f2_sm not in sm1_d:
            #     print("noooo")
            #     return False

        for s in sm1_d:
            if s not in sm2_d:
                return False

        return True
    except Exception as e:
        print(e)
        print(f"except for {sm1,sm2}")
        return False


def check_smiles_fragments(smile_set):
    sm1, sm2 = smile_set
    try:
        m1 = Chem.MolFromSmiles(sm1)
        m1 = mdis.Disconnect(m1)
        sm1_d = []
        for f in Chem.GetMolFrags(m1, asMols=True):
            sm1_d += [Chem.MolToSmiles(f)]
        m2 = Chem.MolFromSmiles(sm2)
        m2 = mdis.Disconnect(m2)
        sm2_d = []
        for f in Chem.GetMolFrags(m2, asMols=True):
            sm2_d += [Chem.MolToSmiles(f)]
        if set(sm1_d) == set(sm2_d):
            return True
        else:
            return False
    except Exception as e:
        print(f"except for {sm1,sm2}")
        print(e)
        return False


def check_smiles_tmc_resonance(smile_set):
    """Checks the resonance overlap between two TMC smiles.

    The start SMILES have to overlap in the two sets
    """
    sm1, sm2 = smile_set

    if Chem.CanonSmiles(sm1) == Chem.CanonSmiles(sm2):
        return True
    try:
        m1 = Chem.MolFromSmiles(sm1)
        m2 = Chem.MolFromSmiles(sm2)

        sm1_d = [
            Chem.CanonSmiles(Chem.MolToSmiles(x))
            for x in Chem.ResonanceMolSupplier(m1, flags=Chem.ALLOW_CHARGE_SEPARATION)
        ]
        sm2_d = [
            Chem.CanonSmiles(Chem.MolToSmiles(x))
            for x in Chem.ResonanceMolSupplier(m2, flags=Chem.ALLOW_CHARGE_SEPARATION)
        ]

        if set(sm1_d) == set(sm2_d):
            return True
        if (sm1 in sm2_d) and (sm2 in sm1_d):
            return True

        return False
    except Exception:
        print(f"except for {sm1,sm2}")
        return False


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
            if i % 10000 == 0:
                print(f"Status: {i} iterations")
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
                print("function raised %s" % error)
                res.append(None)
                # print(error.traceback)  # Python's traceback of remote process
    return res


def fix_NO2(smiles):
    # change to not cound Oxygens bount to tm
    if smiles != smiles:
        return None
    m = Chem.MolFromSmiles(smiles)
    if not m:
        return np.nan
    emol = Chem.RWMol(m)
    patt = Chem.MolFromSmarts(
        "[#8-]-[#7+0]-[#8-].[#21,#22,#23,#24,#25,#26,#27,#28,#29,#30,#39,#40,#41,#42,#43,#44,#45,#46,#47,#48,#57,#72,#73,#74,#75,#76,#77,#78,#79,#80]"
    )
    matches = emol.GetSubstructMatches(patt)
    for a1, a2, a3, a4 in matches:
        if not emol.GetBondBetweenAtoms(a1, a4) and not emol.GetBondBetweenAtoms(
            a3, a4
        ):
            print(a1, a2, a3, a4)
            tm = emol.GetAtomWithIdx(a4)
            o1 = emol.GetAtomWithIdx(a1)
            n = emol.GetAtomWithIdx(a2)
            # o2 = emol.GetAtomWithIdx(a3)
            tm_charge = tm.GetFormalCharge()
            print("old charge = ", tm_charge)
            new_charge = tm_charge - 2
            tm.SetFormalCharge(new_charge)
            n.SetFormalCharge(+1)
            o1.SetFormalCharge(0)
            emol.RemoveBond(a1, a2)
            emol.AddBond(a1, a2, rdchem.BondType.DOUBLE)

    mol = emol.GetMol()
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))

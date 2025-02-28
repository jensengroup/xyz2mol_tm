# TMC SMILES DATASETS

## Data

The data is obtained from the CSD api.
Entries in the CSD can have counter ions. We assume that the complex of interest is always the heaviest fragment.
If the heaviest fragment does not contain a TM we do not consider it.

##### [tmqmg_smiles.csv](tmqmg_smiles.csv)

- The file containing the SMILES for the tmQMg entries

- `IDs` the CSD identifier.
- `smiles_NBO_DFT_xyz` is the SMILES obtained using the charges from the NBO calculations in tmQMg-L and the xyz files from the tmQMg and tmQMg-L.
- `smiles_huckel_CSD_xyz` contains the SMILES from the huckel approach using the xyz files for the heaviest fragment from the CSD API.
- `smiles_huckel_DFT_xyz` contains the SMILES from the huckel approach using the xyz files from the DFT calculations in tmQMg.
- `smiles_CSD_fix` is the SMILES from the CSD_to_valid_SMILES approach.
- `formula_heaviest_fragment` is the formula of the heaviest fragment for the entry.
- `charge` is the formal charge of the heaviest fragment.



##### [csd_smiles.csv](./csd_smiles.csv)

- The file containing ~220K SMILES of mononuclear TMCs from the CSD
- `IDs` the CSD identifier.
- `smiles_csd_api` is the SMILES for the heaviest fragment of the entry returned by the CSD API.
- `smiles_csd_api_fix` is the SMILES from the CSD_to_valid_SMILES approach.
- `huckel_smiles_CSD_xyz` contains the SMILES from the huckel approach using the xyz files for the heaviest fragment from the CSD API.
- `formula_heaviest_fragment` is the formula of the heaviest fragment for the entry. The formal charge is also stated in the formula.

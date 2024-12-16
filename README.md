# xyz2mol for Transition Metal Complexes

Repo for the preprint: SUPER_AWESOME_PAPER

This is is a working in progress repo where data and code will be gradually updated as the preprint goes through review.

## [SMILES_csvs](./SMILES_csvs/)

Folder with SMILES datasets

## [xyz2mol_tm](./xyz2mol_tm/)

Code for converting TMC xyz files into TMC SMILES.

#### [huckel_to_smiles](./xyz2mol_tm/huckel_to_smiles/)

Code for the Huckel to SMILES approach.
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/grass.png)

#### [NBO_to_smiles](./xyz2mol_tm/NBO_to_smiles/)

Code for the NBO to SMILES approach.
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/grass.png)

#### [CSD_to_valid_smiles](./xyz2mol_tm/CSD_to_valid_smiles/)

Code for the CSD SMILES fix approach.
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/grass.png)

## [comparing_smiles](./comparing_smiles/)

Contains two notebooks comparing the three SMILES sets and highlighting different issues with parts of the SMILES sets.

**Notebook comparing the three SMILES sets:**

[comparing_smiles.ipynb](./comparing_smiles/comparing_smiles.ipynb)

**Notebook going through important issues and differences in the SMILES sets:**

[highlight_smiles_problems.ipynb](./comparing_smiles/highlight_smiles_problems.ipynb)

## [ML](./ML/)

All code used for Machine Learning experiments on the SMILES sets.

## ⚙️ Setup environment:

We provide to separate conda environment files that reflect the environments that were used to perform the xyz to SMILES and ML approaches respectively.

The conda environment used for creating the SMILES has fewer dependencies and is found at: [xyz2mol_tmc.yml](./xyz2mol_tm/xyz2mol_tmc.yml).

The ML environment file : [ml_environment.yml](./ML/ml_environment.yml).\
Key dependencies are:

- RDKit
- pytorch
- pytorch_geometric
- scikit-learn
- lightgbm
- optuna
- wandb
- scipy
- numpy

Beware that conda environments with pytorch and pytorch_geometric can have issues with detecting a GPU and it is possible that other versions of the packages are needed if installed on other hardware due to mismatch in cuda versions / drivers. The given .yaml file is exported from an environment that worked with our hardware setup.

After a conda environment has been setup based on one of the .yml file, install the modules in this repo into the environment by running the following from this directory:

```
pip install -e .
```

For instructions how to run individual scripts see the corresponding README files in the different modules.

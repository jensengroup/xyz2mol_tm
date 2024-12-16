# ML on TMC SMILES

This folder contains the code used for ML experiments. This is the code for running ML models and it is based on a clone of the ML code in the [tmQMg ML code repo](https://github.com/uiocompcat/tmQMg/tree/main/scripts/Gilmer-MPNN).

In addition to the NBO-graph based Gilmer models in the [tmQMg ML repo](https://github.com/uiocompcat/tmQMg/tree/main/scripts/Gilmer-MPNN) this folder contains our fingerprint and SMILES graph based models.

## How to use

There are 3 main driver scripts. See the argparser in each script for possible input parameters.

### Examples

Fingerprint models:

```
ml_fingerprint.py
```

SMILES graph models:

```
ml_smiles_graph.py
```

To run the NBO-graph based Gilmer model from [tmQMg](https://github.com/uiocompcat/tmQMg/tree/main/scripts/Gilmer-MPNN) use

```
ml_nbo.py
```

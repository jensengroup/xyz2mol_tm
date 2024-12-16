"""Ml module from:

https://github.com/uiocompcat/tmQMg/tree/main/scripts/Gilmer-MPNN Modified to
our approach.
"""

import numpy as np
import pandas as pd
import scipy as sp
import tools
import torch
import wandb
from HyDGL.element_look_up_table import ElementLookUpTable
from nets import GilmerNetGraphLevelFeatures
from plot import (
    plot_correlation,
    plot_error_by_metal_center_group,
    plot_metal_center_group_histogram,
    plot_target_histogram,
    wandb_plot_error_by_metal_center_group,
)
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from trainer import Trainer
from util import TARGET_UNITS

from tmQMg import tmQMg


def run_ml(
    hyper_param: dict,
    wandb_project_name: str = "TMC-ML-SMILES",
    wandb_entity: str = "strandgaard96-university-of-copenhagen",
):
    # wandb.config = hyper_param
    wandb.init(config=hyper_param, project=wandb_project_name, entity=wandb_entity)

    # set name
    wandb.run.name = hyper_param["name"]

    # set seed
    tools.set_global_seed(hyper_param["seed"])

    # Unpack target_units
    target_unit = hyper_param["data"]["target_unit"]

    # setup data set
    dataset: tmQMg = hyper_param["data"]["dataset"](
        root=hyper_param["data"]["root_dir"],
        graph_type=hyper_param["data"]["graph_representation"],
        targets=hyper_param["data"]["targets"],
        exclude=hyper_param["data"]["outliers"],
        developer_mode=False,
    )
    # obtain dictionary of meta data information
    meta_data_dict = dataset.get_meta_data_dict()

    with open(
        "./test_set_ids/huckel_test_set.txt",
        "r",
    ) as f:
        test_ids = f.read().splitlines()

    test_indices = []
    for i, elem in enumerate(dataset):
        id = elem.id
        if id in test_ids:
            test_indices.append(i)

    test_set = Subset(dataset, test_indices)
    remainder_set = Subset(
        dataset, [i for i in range(0, len(dataset)) if i not in test_indices]
    )
    # divide into subsets
    sets = torch.utils.data.random_split(
        remainder_set,
        [
            len(remainder_set)
            - round(hyper_param["data"]["val_set_size"] * len(dataset)),
            round(hyper_param["data"]["val_set_size"] * len(dataset)),
        ],
    )

    sets.append(test_set)

    print(
        "Using "
        + str(len(dataset))
        + " data points. (train="
        + str(len(sets[0]))
        + ", val="
        + str(len(sets[1]))
        + ", test="
        + str(len(sets[2]))
        + ")"
    )

    # code for linear baseline fit
    offset_dict = None
    if hyper_param["atomic_contribution_linear_fit"]:
        print("Performing atomic contribution linear fit..")

        # set up train element count matrix and train target vector
        train_element_count_matrix = np.zeros((len(sets[0]), 86))
        train_target_vector = np.zeros(((len(sets[0]), len(sets[0][0].y))))

        # obtain data
        for i, mol in enumerate(sets[0]):
            train_target_vector[i] = mol.y
            for element in meta_data_dict[mol.id]["element_counts"].keys():
                train_element_count_matrix[i][
                    ElementLookUpTable.get_atomic_number(element) - 1
                ] = meta_data_dict[mol.id]["element_counts"][element]

        # estimate atomic contributions via linear fit
        E_vector = np.dot(
            sp.linalg.pinv(train_element_count_matrix), train_target_vector
        )

        # set up dictionary to store ID - Offset pairs to apply when producing results
        offset_dict = {}

        # iterate through the whole dataset and subtract the corresponding predicted atomic contributions from the target value
        for subset in sets:
            for i, mol in enumerate(subset):
                element_count_vector = np.zeros((86, 1))

                for element in meta_data_dict[mol.id]["element_counts"].keys():
                    element_count_vector[
                        ElementLookUpTable.get_atomic_number(element) - 1
                    ] = meta_data_dict[mol.id]["element_counts"][element]

                offset = float(np.dot(element_count_vector.T, E_vector))
                offset_dict[mol.id] = offset
                mol.y -= offset

    print("Scaling targets..")
    # obtain matrices for features and targets of the train set
    train_feature_matrix_dict = tools.get_feature_matrix_dict(
        sets[0], hyper_param["scaling"]["features_to_scale"]
    )
    # scale all sets according to train set feature matrices
    for subset in sets:
        if hyper_param["scaling"]["type"] == "standard":
            subset = tools.standard_scale_dataset(subset, train_feature_matrix_dict)

            # if targets are scaled, retrieve means and stds to reconstruct real errors
            if "y" in hyper_param["scaling"]["features_to_scale"]:
                train_target_means = tools.get_feature_means_from_feature_matrix_dict(
                    train_feature_matrix_dict, "y"
                )
                train_target_stds = tools.get_feature_stds_from_feature_matrix_dict(
                    train_feature_matrix_dict, "y"
                )
        else:
            raise ValueError("Scaling type not recognized.")

    # set the size of mini batches (for gradient accumulation)
    if hyper_param["batch_size"] % hyper_param["gradient_accumulation_splits"] == 0:
        mini_batch_size = int(
            hyper_param["batch_size"] / hyper_param["gradient_accumulation_splits"]
        )
    else:
        raise ValueError(
            "Cannot divide batch of length "
            + str(hyper_param["batch_size"])
            + " into "
            + str(hyper_param["gradient_accumulation_splits"])
            + "mini-batches."
        )

    # set up dataloaders
    train_loader = DataLoader(
        sets[0],
        batch_size=mini_batch_size,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
    )
    train_loader_unshuffled = DataLoader(
        sets[0],
        batch_size=mini_batch_size,
        shuffle=False,
        num_workers=32,
        pin_memory=True,
    )
    val_loader = DataLoader(
        sets[1],
        batch_size=mini_batch_size,
        shuffle=False,
        num_workers=32,
        pin_memory=True,
    )
    test_loader = DataLoader(
        sets[2],
        batch_size=mini_batch_size,
        shuffle=False,
        num_workers=32,
        pin_memory=True,
    )

    # set up model
    model = hyper_param["model"]["method"](**hyper_param["model"]["parameters"])
    # set up optimizer and scheduler
    optimizer = hyper_param["optimizer"]["method"](
        model.parameters(), **hyper_param["optimizer"]["parameters"]
    )
    scheduler = hyper_param["scheduler"]["method"](
        optimizer, **hyper_param["scheduler"]["parameters"]
    )

    # run
    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        gradient_accumulation_splits=hyper_param["gradient_accumulation_splits"],
    )
    print("Starting training..")
    trainer.run(
        train_loader,
        train_loader_unshuffled,
        val_loader,
        test_loader,
        n_epochs=hyper_param["n_epochs"],
        target_means=train_target_means,
        target_stds=train_target_stds,
        target_offset_dict=offset_dict,
    )

    # get training set predictions and ground truths
    train_predicted_values = []
    train_true_values = []
    train_ids = []
    train_metal_center_groups = []
    for batch in train_loader:
        train_true_values.extend(
            tools.get_target_list_from_batch(
                batch,
                target_means=train_target_means,
                target_stds=train_target_stds,
                target_offset_dict=offset_dict,
            )
        )
        train_predicted_values.extend(
            trainer.predict_batch(
                batch,
                target_means=train_target_means,
                target_stds=train_target_stds,
                target_offset_dict=offset_dict,
            )
        )
        train_ids.extend(batch.id)
        train_metal_center_groups.extend(
            [meta_data_dict[id]["metal_center_group"] for id in batch.id]
        )

    # get validation set predictions and ground truths
    val_predicted_values = []
    val_true_values = []
    val_ids = []
    val_metal_center_groups = []
    for batch in val_loader:
        val_true_values.extend(
            tools.get_target_list_from_batch(
                batch,
                target_means=train_target_means,
                target_stds=train_target_stds,
                target_offset_dict=offset_dict,
            )
        )
        val_predicted_values.extend(
            trainer.predict_batch(
                batch,
                target_means=train_target_means,
                target_stds=train_target_stds,
                target_offset_dict=offset_dict,
            )
        )
        val_ids.extend(batch.id)
        val_metal_center_groups.extend(
            [meta_data_dict[id]["metal_center_group"] for id in batch.id]
        )

    # get test set predictions and ground truths
    test_predicted_values = []
    test_true_values = []
    test_ids = []
    test_metal_center_groups = []
    for batch in test_loader:
        test_true_values.extend(
            tools.get_target_list_from_batch(
                batch,
                target_means=train_target_means,
                target_stds=train_target_stds,
                target_offset_dict=offset_dict,
            )
        )
        test_predicted_values.extend(
            trainer.predict_batch(
                batch,
                target_means=train_target_means,
                target_stds=train_target_stds,
                target_offset_dict=offset_dict,
            )
        )
        test_ids.extend(batch.id)
        test_metal_center_groups.extend(
            [meta_data_dict[id]["metal_center_group"] for id in batch.id]
        )

    # log predictions

    train_df = pd.DataFrame(
        {
            "id": train_ids,
            "predicted": train_predicted_values,
            "truth": train_true_values,
        }
    )
    wandb.log({"train-predictions": wandb.Table(dataframe=train_df)})

    val_df = pd.DataFrame(
        {"id": val_ids, "predicted": val_predicted_values, "truth": val_true_values}
    )
    wandb.log({"val-predictions": wandb.Table(dataframe=val_df)})

    test_df = pd.DataFrame(
        {"id": test_ids, "predicted": test_predicted_values, "truth": test_true_values}
    )
    wandb.log({"test-predictions": wandb.Table(dataframe=test_df)})

    # log plots

    tmp_file_path = "/tmp/image.png"

    plot_metal_center_group_histogram(
        sets[0], sets[1], sets[2], meta_data_dict, file_path=tmp_file_path
    )
    wandb.log(
        {"Metal center group distribution among sets": wandb.Image(tmp_file_path)}
    )

    plot_correlation(
        train_predicted_values,
        train_true_values,
        file_path=tmp_file_path,
        target_unit=target_unit,
    )
    wandb.log({"Training set prediction correlation": wandb.Image(tmp_file_path)})

    plot_correlation(
        val_predicted_values,
        val_true_values,
        file_path=tmp_file_path,
        target_unit=target_unit,
    )
    wandb.log({"Validation set prediction correlation": wandb.Image(tmp_file_path)})

    plot_correlation(
        test_predicted_values,
        test_true_values,
        file_path=tmp_file_path,
        target_unit=target_unit,
    )
    wandb.log({"Test set prediction correlation": wandb.Image(tmp_file_path)})

    plot_target_histogram(
        train_true_values,
        test_true_values,
        file_path=tmp_file_path,
    )
    wandb.log({"Target value distributions": wandb.Image(tmp_file_path)})

    plot_error_by_metal_center_group(
        test_predicted_values,
        test_true_values,
        test_metal_center_groups,
        file_path=tmp_file_path,
    )
    wandb.log({"Test set error by metal center group": wandb.Image(tmp_file_path)})

    wandb.log(
        {
            "test_set_error_by_metal": wandb_plot_error_by_metal_center_group(
                test_predicted_values, test_true_values, test_metal_center_groups
            )
        }
    )

    # end run
    wandb.finish(exit_code=0)


def run_baseline(target: str, use_atomic_contribution_linear_fit: bool):
    with open("./../tmQMg/outliers.txt", "r") as fh:
        outliers = fh.read().splitlines()

    hyper_param = {
        "name": "Baseline - " + target,
        "data": {
            "dataset": tmQMg,
            "root_dir": "root_dir",
            "val_set_size": 0.1,
            "test_set_size": 0.1,
            "graph_representation": "baseline",
            "targets": [target],
            "outliers": outliers,
            "target_unit": TARGET_UNITS[target],
        },
        "model": {
            "name": "GilmerNet",
            "method": GilmerNetGraphLevelFeatures,
            "parameters": {
                "n_node_features": 4,
                "n_edge_features": 2,
                "n_graph_features": 4,
                "dim": 128,
                "set2set_steps": 4,
                "n_atom_jumps": 4,
            },
        },
        "optimizer": {
            "name": "Adam",
            "method": torch.optim.Adam,
            "parameters": {"lr": 0.001},
        },
        "scheduler": {
            "name": "ReduceLrOnPlateau",
            "method": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "parameters": {
                "mode": "min",
                "factor": 0.7,
                "patience": 5,
                "min_lr": 0.00001,
            },
        },
        "scaling": {
            "type": "standard",
            "features_to_scale": ["x", "edge_attr", "graph_attr", "y"],
        },
        "atomic_contribution_linear_fit": use_atomic_contribution_linear_fit,
        "batch_size": 32,
        "gradient_accumulation_splits": 1,
        "n_epochs": 300,
        "seed": 2022,
    }

    run_ml(hyper_param)


def run_uNatQ(target: str, use_atomic_contribution_linear_fit: bool):
    with open("./../tmQMg/outliers.txt", "r") as fh:
        outliers = fh.read().splitlines()

    hyper_param = {
        "name": "u-NatQ - " + target,
        "data": {
            "dataset": tmQMg,
            "root_dir": "root_dir",
            "val_set_size": 0.1,
            "test_set_size": 0.1,
            "graph_representation": "uNatQ",
            "targets": [target],
            "outliers": outliers,
            "target_unit": TARGET_UNITS[target],
        },
        "model": {
            "name": "GilmerNet",
            "method": GilmerNetGraphLevelFeatures,
            "parameters": {
                "n_node_features": 21,
                "n_edge_features": 19,
                "n_graph_features": 4,
                "dim": 128,
                "set2set_steps": 4,
                "n_atom_jumps": 4,
            },
        },
        "optimizer": {
            "name": "Adam",
            "method": torch.optim.Adam,
            "parameters": {"lr": 0.001},
        },
        "scheduler": {
            "name": "ReduceLrOnPlateau",
            "method": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "parameters": {
                "mode": "min",
                "factor": 0.7,
                "patience": 5,
                "min_lr": 0.00001,
            },
        },
        "scaling": {
            "type": "standard",
            "features_to_scale": ["x", "edge_attr", "graph_attr", "y"],
        },
        "atomic_contribution_linear_fit": use_atomic_contribution_linear_fit,
        "batch_size": 32,
        "gradient_accumulation_splits": 1,
        "n_epochs": 300,
        "seed": 2022,
    }

    run_ml(hyper_param)


def run_dNatQ(target: str, use_atomic_contribution_linear_fit: bool):
    with open("./../tmQMg/outliers.txt", "r") as fh:
        outliers = fh.read().splitlines()

    hyper_param = {
        "name": "d-NatQ - " + target,
        "data": {
            "dataset": tmQMg,
            "root_dir": "root_dir",
            "val_set_size": 0.1,
            "test_set_size": 0.1,
            "graph_representation": "dNatQ",
            "targets": [target],
            "outliers": outliers,
            "target_unit": TARGET_UNITS[target],
        },
        "model": {
            "name": "GilmerNet",
            "method": GilmerNetGraphLevelFeatures,
            "parameters": {
                "n_node_features": 21,
                "n_edge_features": 26,
                "n_graph_features": 4,
                "dim": 128,
                "set2set_steps": 4,
                "n_atom_jumps": 6,
            },
        },
        "optimizer": {
            "name": "Adam",
            "method": torch.optim.Adam,
            "parameters": {"lr": 0.001},
        },
        "scheduler": {
            "name": "ReduceLrOnPlateau",
            "method": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "parameters": {
                "mode": "min",
                "factor": 0.7,
                "patience": 5,
                "min_lr": 0.00001,
            },
        },
        "scaling": {
            "type": "standard",
            "features_to_scale": ["x", "edge_attr", "graph_attr", "y"],
        },
        "atomic_contribution_linear_fit": use_atomic_contribution_linear_fit,
        "batch_size": 32,
        "gradient_accumulation_splits": 1,
        "n_epochs": 300,
        "seed": 2022,
    }

    run_ml(hyper_param)


# - - - entry point - - - #
if __name__ == "__main__":
    # general targets
    targets = [
        "tzvp_homo_lumo_gap",
        "polarisability",
        "tzvp_dipole_moment",
    ]

    for target in targets:
        run_baseline(target, False)
        run_uNatQ(target, False)
        run_dNatQ(target, False)

import argparse
import os

import numpy as np
import pandas as pd
import torch
import wandb
from load_smiles_data import load_data
from nets import MolecularNN
from plot import plot_correlation, plot_target_histogram
from sklearn.ensemble import RandomForestRegressor
from trainer import HyperParameterGuider, Trainer_fingerprint
from util import (
    PATH_TO_TMQMG,
    TARGET_UNITS,
    get_dataset_path,
    mae,
    map_smiles_column_name,
    r_squared,
    rmse,
    smiles2fp,
)


def run_ml_hyperparam(
    hyper_param: dict,
    wandb_project_name: str = "TMC-ML-SMILES",
    wandb_entity: str = "strandgaard96-university-of-copenhagen",
):
    job_type = "developer_mode" if not args.developer_mode else "production"
    # wandb.config = hyper_param
    wandb.init(
        config=hyper_param,
        project=wandb_project_name,
        entity=wandb_entity,
        group="Production",
        job_type=job_type,
    )

    # set name
    wandb.run.name = hyper_param["name"]

    # set seed
    np.random.seed(hyper_param["seed"])

    # Unpack target unit
    target_unit = hyper_param["data"]["target_unit"]

    # Get the dataframe column label based on the chosen SMILE set
    smiles_column_string = map_smiles_column_name(chosen_smiles=args.smiles_method)

    with open(
        "./test_set_ids/test_set_kneiding.txt",
        "r",
    ) as f:
        test_ids = f.read().splitlines()

    # construct the training data from the smiles data and targets.
    df_training, df_test = load_data(
        hyper_param["data"]["targets_path"],
        hyper_param["data"]["smiles_path"],
        test_ids=test_ids,
        outlier_path=hyper_param["data"]["outlier_path"],
        smiles_column_string=smiles_column_string,
        chirality=args.chirality,
    )
    if args.developer_mode:
        df_training = df_training[0:1000]
        df_test = df_test[0:1000]

    fps_training = [
        smiles2fp(
            smiles, hyper_param["model"]["encoding"], hyper_param["data"]["chirality"]
        )
        for smiles in df_training[smiles_column_string]
    ]
    fps_test = [
        smiles2fp(
            smiles, hyper_param["model"]["encoding"], hyper_param["data"]["chirality"]
        )
        for smiles in df_test[smiles_column_string]
    ]
    df_training.loc[:, "fingerprints"] = fps_training
    df_test.loc[:, "fingerprints"] = fps_test

    # Define training features and targets
    training_features = df_training["fingerprints"].values
    training_features = np.stack(training_features)
    training_targets = df_training[hyper_param["data"]["targets"]].values.squeeze()
    test_features = df_test["fingerprints"].values
    test_features = np.stack(test_features)
    test_targets = df_test[hyper_param["data"]["targets"]].values.squeeze()
    #

    print("Starting gridsearch..")
    hyper = HyperParameterGuider(hyper_param)
    model = hyper.optimize_hyperparameters(training_features, training_targets)
    print("Finished gridsearch")
    # get training set predictions and ground truths
    training_prediction = model.predict(training_features)
    test_prediction = model.predict(test_features)

    # log predictions
    df_training["predicted_values"] = training_prediction
    df_test["predicted_values"] = test_prediction

    error_metrics = {
        "train_mae": mae(training_prediction, training_targets),
        "test_mae": mae(test_prediction, test_targets),
        "train_rmse": rmse(training_prediction, training_targets),
        "test_rmse": rmse(test_prediction, test_targets),
        "train_r_squared": r_squared(training_prediction, training_targets),
        "test_r_squared": r_squared(test_prediction, test_targets),
    }

    print(
        f"R²: Training - {r_squared(training_prediction, training_targets)} Test - {r_squared(test_prediction, test_targets)}"
    )

    tmp_file_path = "/tmp/image.png"
    wandb.log(error_metrics)

    test_df = pd.DataFrame(
        {"id": df_test.IDs, "predicted": test_prediction, "truth": test_targets}
    )
    wandb.log({"test-predictions": wandb.Table(dataframe=test_df)})
    train_df = pd.DataFrame(
        {
            "id": df_training.IDs,
            "predicted": training_prediction,
            "truth": training_targets,
        }
    )
    wandb.log({"train-predictions": wandb.Table(dataframe=train_df)})

    plot_correlation(
        training_prediction,
        training_targets,
        file_path=tmp_file_path,
        target_unit=target_unit,
    )
    wandb.log({"Training set prediction correlation": wandb.Image(tmp_file_path)})

    plot_correlation(
        test_prediction, test_targets, file_path=tmp_file_path, target_unit=target_unit
    )
    wandb.log({"Test set prediction correlation": wandb.Image(tmp_file_path)})

    plot_target_histogram(training_targets, test_targets, file_path=tmp_file_path)
    wandb.log({"Target value distributions": wandb.Image(tmp_file_path)})

    # end run
    wandb.finish(exit_code=0)


def run_ml(
    hyper_param: dict,
    wandb_project_name: str = "TMC-ML-SMILES",
    wandb_entity: str = "strandgaard96-university-of-copenhagen",
):
    """Driver for the fingerprint models.

    Args:
        hyper_param: Specifies are relevant parameters
        wandb_project_name: WANDB project name tied to the entity account
        wandb_entity: Name of the entity in WANDB
    """
    job_type = "developer_mode" if not args.developer_mode else "production"
    # wandb.config = hyper_param
    wandb.init(
        config=hyper_param,
        project=wandb_project_name,
        entity=wandb_entity,
        group="Fingerprints",
        job_type=job_type,
    )

    # set name
    wandb.run.name = hyper_param["name"]

    # set seed
    np.random.seed(hyper_param["seed"])

    # Unpack target unit
    target_unit = hyper_param["data"]["target_unit"]

    # Get the dataframe column label based on the chosen SMILE set
    smiles_column_string = map_smiles_column_name(chosen_smiles=args.smiles_method)

    with open(
        "./test_set_ids/test_set_kneiding.txt",
        "r",
    ) as f:
        test_ids = f.read().splitlines()

    df_training, df_test = load_data(
        hyper_param["data"]["targets_path"],
        hyper_param["data"]["smiles_path"],
        test_ids=test_ids,
        outlier_path=hyper_param["data"]["outlier_path"],
        smiles_column_string=smiles_column_string,
        chirality=args.chirality,
    )

    if args.developer_mode:
        df_training = df_training[0:1000]
        df_test = df_test[0:1000]

    fps_training = [
        smiles2fp(
            smiles, hyper_param["model"]["encoding"], hyper_param["data"]["chirality"]
        )
        for smiles in df_training[smiles_column_string]
    ]
    fps_test = [
        smiles2fp(
            smiles, hyper_param["model"]["encoding"], hyper_param["data"]["chirality"]
        )
        for smiles in df_test[smiles_column_string]
    ]
    df_training.loc[:, "fingerprints"] = fps_training
    df_test.loc[:, "fingerprints"] = fps_test

    # Define training features and targets
    training_features = df_training["fingerprints"].values
    training_features = np.stack(training_features)
    training_features = training_features.astype(float)
    training_targets = df_training[hyper_param["data"]["targets"]].values.squeeze()

    test_features = df_test["fingerprints"].values
    test_features = np.stack(test_features)
    test_features = test_features.astype(float)
    test_targets = df_test[hyper_param["data"]["targets"]].values.squeeze()

    trainer = Trainer_fingerprint(hyper_param)

    print(
        training_features.shape,
        training_targets.shape,
        training_features[0:5],
        training_targets[0:5],
        test_features[0:5],
    )

    # run
    print("Starting training..")
    trainer.fit(training_features, training_targets)
    print("Finished training")

    # Get training set predictions and ground truths
    if hyper_param["model"]["name"] == "NN":
        training_prediction = trainer.trainer_nn.predict(training_features)
        test_prediction = trainer.trainer_nn.predict(test_features)
    else:
        training_prediction = trainer.model.predict(training_features)
        test_prediction = trainer.model.predict(test_features)

    # Store predictions
    df_training["predicted_values"] = training_prediction
    df_test["predicted_values"] = test_prediction

    error_metrics = {
        "train_mae": mae(training_prediction, training_targets),
        "test_mae": mae(test_prediction, test_targets),
        "train_rmse": rmse(training_prediction, training_targets),
        "test_rmse": rmse(test_prediction, test_targets),
        "train_r_squared": r_squared(training_prediction, training_targets),
        "test_r_squared": r_squared(test_prediction, test_targets),
    }

    print(
        f"R²: Training - {r_squared(training_prediction, training_targets)} Test - {r_squared(test_prediction, test_targets)}"
    )

    tmp_file_path = "/tmp/image.png"
    wandb.log(error_metrics)

    test_df = pd.DataFrame(
        {"id": df_test.IDs, "predicted": test_prediction, "truth": test_targets}
    )
    wandb.log({"test-predictions": wandb.Table(dataframe=test_df)})
    train_df = pd.DataFrame(
        {
            "id": df_training.IDs,
            "predicted": training_prediction,
            "truth": training_targets,
        }
    )
    wandb.log({"train-predictions": wandb.Table(dataframe=train_df)})

    plot_correlation(
        training_prediction,
        training_targets,
        file_path=tmp_file_path,
        target_unit=target_unit,
    )
    wandb.log({"Training set prediction correlation": wandb.Image(tmp_file_path)})

    plot_correlation(
        test_prediction, test_targets, file_path=tmp_file_path, target_unit=target_unit
    )
    wandb.log({"Test set prediction correlation": wandb.Image(tmp_file_path)})

    plot_target_histogram(training_targets, test_targets, file_path=tmp_file_path)
    wandb.log({"Target value distributions": wandb.Image(tmp_file_path)})

    # end run
    wandb.finish(exit_code=0)


def run_forest(params):
    target, encoding = params

    # Define the path for the smiles dataset
    smiles_path = get_dataset_path(args.dataset)
    print(smiles_path)

    if args.hyperparam_opt:
        name = "RandomForest-gridsearch"
        func = run_ml_hyperparam
    else:
        name = "RandomForest"
        func = run_ml

    hyper_param = {
        "name": f"{name} - " + target,
        "data": {
            "targets_path": PATH_TO_TMQMG / "tmQMg_properties_and_targets.csv",
            "smiles_path": smiles_path,
            "targets": [target],
            "outlier_path": PATH_TO_TMQMG / "outliers.txt",
            "target_unit": TARGET_UNITS[target],
            "chirality": args.chirality,
        },
        "model": {
            "name": name,
            "method": RandomForestRegressor,
            "parameters": {
                "n_estimators": 200,
                "min_samples_leaf": 2,
                "n_jobs": args.njobs,
                "verbose": 1,
            },
            "encoding": encoding,
        },
        "seed": 2024,
        "commandline_args": vars(args),
    }

    func(hyper_param)


def run_lgbm(params):
    target, encoding = params

    # Define the path for the smiles dataset
    smiles_path = get_dataset_path(args.dataset)

    if args.hyperparam_opt:
        func = run_ml_hyperparam
        name = "LightGbm-hyperopt"
    else:
        func = run_ml
        name = "LightGbm"

    hyper_param = {
        "name": f"{name} - " + target,
        "data": {
            "targets_path": PATH_TO_TMQMG / "tmQMg_properties_and_targets.csv",
            "smiles_path": smiles_path,
            "targets": [target],
            "outlier_path": PATH_TO_TMQMG / "outliers.txt",
            "target_unit": TARGET_UNITS[target],
            "chirality": args.chirality,
        },
        "model": {
            "name": name,
            "method": "LightGbm",
            "parameters": {},
            "encoding": encoding,
        },
        "seed": 2024,
        "commandline_args": vars(args),
    }

    func(hyper_param)


def run_nn(params):
    target, encoding = params

    # Define the path for the smiles dataset
    smiles_path = get_dataset_path(args.dataset)

    if args.hyperparam_opt:
        func = run_ml_hyperparam
        name = "NN-hyperopt"
    else:
        func = run_ml
        name = "NN"

    hyper_param = {
        "name": f"{name} - " + target,
        "data": {
            "targets_path": PATH_TO_TMQMG / "tmQMg_properties_and_targets.csv",
            "smiles_path": smiles_path,
            "targets": [target],
            "outlier_path": PATH_TO_TMQMG / "outliers.txt",
            "val_set_size": 0.1,
            "target_unit": TARGET_UNITS[target],
            "chirality": args.chirality,
        },
        "model": {
            "name": name,
            "method": MolecularNN,
            "parameters": {"input_size": 1024, "hidden_size": 256, "output_size": 1},
            "encoding": encoding,
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
                "min_lr": 0.001,
            },
        },
        "seed": 2024,
        "commandline_args": vars(args),
        "batch_size": 32,
        "n_epochs": 300,
    }

    func(hyper_param)


def get_arguments(arg_list=None) -> argparse.Namespace:
    """

    Args:
        arg_list: Automatically obtained from the commandline if provided.
        Otherwise default arguments are used

    Returns:
        parser.parse_args(arg_list)(Namespace): Dictionary like class that contain the driver arguments.

    """
    parser = argparse.ArgumentParser(
        description="Run fingerprint based ML models on SMILES",
        fromfile_prefix_chars="+",
    )
    parser.add_argument(
        "--smiles_method",
        type=str,
        choices=["nbo", "csd", "huckel"],
        default="huckel",
        help="Sets which of the 3 SMILES sets to use labeled by the respective methods used to get the SMILES",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["tmqmg", "csd"],
        default="tmqmg",
        help="Select which set to get SMILES from. The tmQMg sets or the large ~220K CSD set.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["randomforest", "lgbm", "nn"],
        default="randomforest",
        help="Select which ml model to train with",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Number of cross validations used in randomforest",
    )
    parser.add_argument(
        "--njobs",
        type=int,
        default=6,
        help="Paralellization option for data processing and cpu models",
    )
    parser.add_argument(
        "--developer_mode",
        action="store_true",
        help="Used to only take a sample of the input data for debugging.",
    )
    parser.add_argument(
        "--chirality",
        action="store_true",
        help="Whether to include chirality in fingerprint encodings",
    )
    parser.add_argument(
        "--hyperparam_opt",
        action="store_true",
        help="Activates hyperparameter optimization of the chosen model and SMILES set.",
    )
    return parser.parse_args(arg_list)


# - - - entry point - - - #
if __name__ == "__main__":
    args = get_arguments()
    if args.developer_mode:
        os.environ["WANDB_MODE"] = "offline"

    # Setup targets and encoding
    targets = [
        ("polarisability", "count"),
        ("tzvp_homo_lumo_gap", "count"),
        ("tzvp_dipole_moment", "count"),
    ]
    func_mapper = {"randomforest": run_forest, "lgbm": run_lgbm, "nn": run_nn}

    for target in targets:
        func_mapper[args.method](target)

    # end run
    wandb.finish(exit_code=0)

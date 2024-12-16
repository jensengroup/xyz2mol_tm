import argparse
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch_geometric
import wandb
from encode_graph import create_pytorch_geometric_graph_data_list_from_smiles_and_labels
from gcnn_tmc import GraphLevelGNN
from load_smiles_data import load_data
from nets import Wrapper_GilmerNet
from plot import plot_correlation, plot_target_histogram
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from util import (
    PATH_TO_TMQMG,
    TARGET_UNITS,
    get_dataset_path,
    mae,
    map_smiles_column_name,
    r_squared,
    rmse,
)


def run_graph_ml(
    hyper_param: dict,
    wandb_project_name: str = "TMC-ML-SMILES",
    wandb_entity: str = "strandgaard96-university-of-copenhagen",
):
    """Driver for the SMILES graph models.

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
        group="Graph",
        job_type=job_type,
    )

    # set name
    wandb.run.name = hyper_param["name"]

    # Unpack target units
    target_unit = hyper_param["data"]["target_unit"]

    # set seed
    pl.seed_everything(42)

    with open(
        "./test_set_ids/test_set_kneiding.txt",
        "r",
    ) as f:
        test_ids = f.read().splitlines()

    smiles_column_string = map_smiles_column_name(chosen_smiles=args.smiles_method)

    df_training, df_test = load_data(
        hyper_param["data"]["targets_path"],
        hyper_param["data"]["smiles_path"],
        test_ids=test_ids,
        outlier_path=hyper_param["data"]["outlier_path"],
        smiles_column_string=smiles_column_string,
        chirality=hyper_param["data"]["chirality"],
    )

    if args.developer_mode:
        df_training = df_training[0:1000]
        df_test = df_test[0:200]

    train_smiles = df_training[smiles_column_string]
    train_target = np.array(df_training[hyper_param["data"]["targets"][0]])
    test_smiles = df_test[smiles_column_string]
    test_target = np.array(df_test[hyper_param["data"]["targets"][0]])

    training_ids = df_training.IDs
    test_ids = df_test.IDs
    print(training_ids, test_ids)

    data_list_train = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(
        train_smiles,
        train_target,
        ids=training_ids,
        use_chirality=hyper_param["data"]["chirality"],
    )
    data_list_test = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(
        test_smiles,
        test_target,
        ids=test_ids,
        use_chirality=hyper_param["data"]["chirality"],
    )

    # Split the train list into train and validation
    data_list_train, data_list_val = train_test_split(
        data_list_train, test_size=0.1, shuffle=False, random_state=42
    )

    model, trainer, result = train_graph_classifier(
        model=hyper_param["model"]["name"],
        train_data=data_list_train,
        validation_data=data_list_val,
    )

    graph_train_loader = torch_geometric.loader.DataLoader(
        dataset=data_list_train, batch_size=4, num_workers=7
    )
    graph_val_loader = torch_geometric.loader.DataLoader(
        dataset=data_list_val, batch_size=4, num_workers=7
    )
    graph_test_loader = torch_geometric.loader.DataLoader(
        dataset=data_list_test, batch_size=4, num_workers=7
    )

    model.eval()

    # Evaluate the trained model on the three sets
    train_preds = []
    for k, batch in enumerate(graph_train_loader):
        # compute current value of loss function via forward pass
        _, preds = model.predict_step(batch, k)
        preds = preds.cpu().detach().numpy().tolist()
        if isinstance(preds, float):
            preds = [preds]
        train_preds.extend(preds)

    val_preds = []
    for k, batch in enumerate(graph_val_loader):
        # compute current value of loss function via forward pass
        _, preds = model.predict_step(batch, k)
        preds = preds.cpu().detach().numpy().tolist()
        if isinstance(preds, float):
            preds = [preds]
        val_preds.extend(preds)

    test_preds = []
    for k, batch in enumerate(graph_test_loader):
        # compute current value of loss function via forward pass
        _, preds = model.predict_step(batch, k)
        preds = preds.cpu().detach().numpy().tolist()
        if isinstance(preds, float):
            preds = [preds]
        test_preds.extend(preds)

    # log predictions
    training_prediction = train_preds
    test_prediction = test_preds
    training_targets = np.array([data.y for data in data_list_train]).flatten()
    test_targets = np.array([data.y for data in data_list_test]).flatten()

    training_ids = np.array([data.identifier for data in data_list_train]).flatten()
    test_ids = np.array([data.identifier for data in data_list_test]).flatten()

    # TODO: What is the null model Maria?
    nullmodel_mae = np.mean(np.absolute(test_targets - np.mean(test_targets)))
    print("null model = ", nullmodel_mae)

    error_metrics = {
        "train_mae": mae(training_prediction, training_targets),
        "test_mae": mae(test_prediction, test_targets),
        "train_rmse": rmse(training_prediction, training_targets),
        "test_rmse": rmse(test_prediction, test_targets),
        "train_r_squared": r_squared(training_prediction, training_targets),
        "test_r_squared": r_squared(test_prediction, test_targets),
        "null_model_mae": nullmodel_mae,
    }

    print(
        f"R²: Training - {r_squared(training_prediction, training_targets)} Test - {r_squared(test_prediction, test_targets)}"
    )

    tmp_file_path = "/tmp/image.png"
    wandb.log(error_metrics)

    plot_correlation(
        training_prediction,
        training_targets.flatten(),
        file_path=tmp_file_path,
        target_unit=target_unit,
    )
    wandb.log({"Training set prediction correlation": wandb.Image(tmp_file_path)})

    plot_correlation(
        test_prediction,
        test_targets.flatten(),
        file_path=tmp_file_path,
        target_unit=target_unit,
    )
    wandb.log({"Test set prediction correlation": wandb.Image(tmp_file_path)})

    plot_target_histogram(
        training_targets, test_targets.flatten(), file_path=tmp_file_path
    )

    wandb.log({"Target value distributions": wandb.Image(tmp_file_path)})

    df_test_summary = pd.DataFrame(
        data={
            "IDs": test_ids,
            "targets": test_targets.flatten(),
            "predictions": test_prediction,
        }
    )
    df_training_summary = pd.DataFrame(
        data={
            "IDs": training_ids,
            "targets": training_targets.flatten(),
            "predictions": training_prediction,
        }
    )
    wandb.log({"train-predictions": wandb.Table(dataframe=df_training_summary)})
    wandb.log({"test-predictions": wandb.Table(dataframe=df_test_summary)})

    # end run
    wandb.finish(exit_code=0)


def train_graph_classifier(model, train_data, validation_data, **model_kwargs):
    """Train a model on the graph data.

    Args:
        model (str): Name of the graph model to use
        train_data : Graph data
        validation_data : Graph validation data
        **model_kwargs: Parameters defining model architechture
    """
    # Extract graph dimensions from the data
    c_in = train_data[0].num_node_features
    c_edge_in = train_data[0].num_edge_features

    graph_train_loader = torch_geometric.loader.DataLoader(
        dataset=train_data, batch_size=8, num_workers=7
    )
    graph_val_loader = torch_geometric.loader.DataLoader(
        dataset=validation_data, batch_size=8, num_workers=7
    )

    # Create a PyTorch Lightning trainer with the generation callback
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    root_dir = "."
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ModelCheckpoint(save_weights_only=True, monitor="val_loss"),
        ],
        accelerator=accelerator,  # gpus=1 if str(device).startswith("cuda") else 0,
        devices=1,
        max_epochs=150,
        logger=WandbLogger(),
    )  # ,
    # progress_bar_refresh_rate=1)
    trainer.logger._default_hp_metric = (
        None  # Optional logging argument that we don't need
    )

    # Run one of the models
    if model == "GCNN":
        model = GraphLevelGNN(
            batch_size=32,
            c_in=c_in,
            c_out=1,
            c_edge_in=c_edge_in,
            c_edge_hidden=30,
            c_hidden=256,
            dp_rate_linear=0.1,
            dp_rate=0.0,
            num_layers=4,
        )
        trainer.fit(model, graph_train_loader, graph_val_loader)
        model = GraphLevelGNN.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )
    elif model == "Gilmer":
        model = Wrapper_GilmerNet(
            batch_size=32,
            n_node_features=c_in,
            n_edge_features=c_edge_in,
            dim=256,
            set2set_steps=4,
            n_atom_jumps=4,
            aggr_function="mean",
        )
        trainer.fit(model, graph_train_loader, graph_val_loader)
        model = Wrapper_GilmerNet.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )
    else:
        raise Exception("Not valid model specified")

    # Test best model on validation and test set
    train_result = trainer.test(model, dataloaders=graph_train_loader, verbose=False)
    val_result = trainer.test(model, dataloaders=graph_val_loader, verbose=False)
    print("validation:", val_result)
    print("train:", train_result)
    result = {"val": val_result[0]["test_loss"], "train": train_result[0]["test_loss"]}
    return model, trainer, result


def run_ml(params):
    target, encoding = params

    # Define the path for the smiles dataset
    smiles_path = get_dataset_path(args.dataset)

    name = args.method
    func = run_graph_ml

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
            "parameters": {},
        },
        "seed": 2024,
        "commandline_args": vars(args),
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
        description="Run graph models on our different smiles datasets",
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
        choices=["GCNN", "Gilmer"],
        default="GCNN",
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

    # general targets and encodings
    targets = [
        ("polarisability", "count"),
        # ("polarisability", "one-hot"),
        ("tzvp_homo_lumo_gap", "count"),
        # ("tzvp_homo_lumo_gap", "one-hot"),
        ("tzvp_dipole_moment", "count"),
        # ("tzvp_dipole_moment", "one-hot"),
    ]

    for target in targets:
        run_ml(target)

    # end run
    wandb.finish(exit_code=0)

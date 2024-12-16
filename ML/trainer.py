import json

import lightgbm as lgbm
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from LGBMRegressor_hyperopt import optimize_lgbm
from sklearn.model_selection import GridSearchCV
from tools import MolecularDataset, get_target_list


class Trainer:
    def __init__(
        self, model, optimizer, scheduler=None, gradient_accumulation_splits=1
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Using device " + str(self.device) + ".")

        self._model = model.to(self.device)
        self._optimizer = optimizer
        self._scheduler = scheduler

        self._gradient_accumulation_splits = gradient_accumulation_splits

    @property
    def model(self):
        """Getter for model."""
        return self._model

    def _train(self, train_loader):
        """Performs a full training step. Depending on the setting for gradient
        accumulation, performs backward pass only every n batch.

        Returns:
            float: The obtained training loss.
        """
        self._model.train()
        loss_all = 0

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            loss = F.mse_loss(self._model(batch), batch.y)
            loss.backward()
            loss_all += loss.item() * batch.num_graphs

            # gradient accumulation
            if ((batch_idx + 1) % self._gradient_accumulation_splits == 0) or (
                batch_idx + 1 == len(train_loader)
            ):
                self._optimizer.step()
                self._optimizer.zero_grad()

        return loss_all / len(train_loader.dataset)

    def _mae(self, predictions, targets):
        errors = np.abs(predictions - targets)
        return np.mean(errors)

    def _median(self, predictions, targets):
        errors = np.abs(predictions - targets)
        return np.median(errors)

    def _rmse(self, predictions, targets):
        errors = np.abs(predictions - targets)
        return np.sqrt(np.mean(np.power(errors, 2)))

    def _r_squared(self, predictions, targets):
        target_mean = np.mean(targets)
        return 1 - (
            np.sum(np.power(targets - predictions, 2))
            / np.sum(np.power(targets - target_mean, 2))
        )

    def predict_batch(
        self, batch, target_means=0, target_stds=1, target_offset_dict=None
    ):
        """Makes predictions on a given batch.

        Returns:
            list: The predictions.
        """

        self._model.eval()
        batch = batch.to(self.device)

        # get data point specific offsets if specified
        offset = 0
        if target_offset_dict is not None:
            offset = np.array([target_offset_dict[i] for i in batch.id])

        # get predictions for batch
        predictions = (
            self.model(batch).cpu().detach().numpy() * target_stds
            + target_means
            + offset
        ).tolist()

        return predictions

    def predict_loader(
        self, loader, target_means=0, target_stds=1, target_offset_dict=None
    ):
        """Makes predictions on a given dataloader.

        Returns:
            list: The predictions.
        """

        predictions = []

        for batch in loader:
            predictions.extend(
                self.predict_batch(
                    batch,
                    target_means=target_means,
                    target_stds=target_stds,
                    target_offset_dict=target_offset_dict,
                )
            )

        return predictions

    def run(
        self,
        train_loader,
        train_loader_unshuffled,
        val_loader,
        test_loader,
        n_epochs=300,
        target_means=0,
        target_stds=1,
        target_offset_dict=None,
    ):
        """Runs a full training loop with automatic metric logging through
        wandb.

        Args:
            train_loader (Dataloader): The dataloader for the training points.
            train_loader_unshuffled (Dataloader): The dataloader for the training points but unshuffled. This is used to calculate metrics on the training set.
            val_loader (Dataloader): The dataloader for the validation points.
            test_loader (Dataloader): The dataloader for the test points.
            n_epochs (int): The number of epochs to perform.
            target_means(np.array): An array of the target means from standard scaling.
            target_stds(np.array): An array of the target stds from standard scaling.
            target_offset_dict (dict): A dictionary that contains ID - Offset pairs that specifies offsets to be added for each individual
                data point. These will be applied when getting targets and predictions.

        Returns:
            model: The trained model.
        """

        # get targets off all sets
        train_targets = get_target_list(
            train_loader_unshuffled,
            target_means=target_means,
            target_stds=target_stds,
            target_offset_dict=target_offset_dict,
        )
        val_targets = get_target_list(
            val_loader,
            target_means=target_means,
            target_stds=target_stds,
            target_offset_dict=target_offset_dict,
        )
        test_targets = get_target_list(
            test_loader,
            target_means=target_means,
            target_stds=target_stds,
            target_offset_dict=target_offset_dict,
        )

        best_val_error = None
        for epoch in range(1, n_epochs + 1):
            # get learning rate from scheduler
            if self._scheduler is not None:
                lr = self._scheduler.optimizer.param_groups[0]["lr"]

            # training step
            loss = self._train(train_loader)

            # get predictions for all sets
            training_preduction = np.array(
                self.predict_loader(
                    train_loader_unshuffled,
                    target_means=target_means,
                    target_stds=target_stds,
                    target_offset_dict=target_offset_dict,
                )
            )
            val_predictions = np.array(
                self.predict_loader(
                    val_loader,
                    target_means=target_means,
                    target_stds=target_stds,
                    target_offset_dict=target_offset_dict,
                )
            )
            test_predictions = np.array(
                self.predict_loader(
                    test_loader,
                    target_means=target_means,
                    target_stds=target_stds,
                    target_offset_dict=target_offset_dict,
                )
            )

            train_error = self._mae(train_targets, training_preduction)
            val_error = self._mae(val_targets, val_predictions)

            # learning rate scheduler step
            if self._scheduler is not None:
                self._scheduler.step(val_error)

            # retain early stop test error
            if best_val_error is None or val_error <= best_val_error:
                test_error = self._mae(test_targets, test_predictions)
                best_val_error = val_error

            output_line = (
                f"Epoch: {epoch:03d}, LR: {lr:7f}, Loss: {loss:.7f}, "
                f"Train MAE: {train_error:.7f}, "
                f"Val MAE: {val_error:.7f}, Test MAE: {test_error:.7f}"
            )
            print(output_line)

            # wandb logging
            wandb.log({"loss": loss}, step=epoch)
            wandb.log({"train_error": train_error}, step=epoch)
            wandb.log({"val_error": val_error}, step=epoch)
            wandb.log({"test_error": test_error}, step=epoch)

            wandb.log(
                {"train_mae": self._mae(training_preduction, train_targets)}, step=epoch
            )
            wandb.log({"val_mae": self._mae(val_predictions, val_targets)}, step=epoch)
            wandb.log(
                {"test_mae": self._mae(test_predictions, test_targets)}, step=epoch
            )

            wandb.log(
                {"train_median": self._median(training_preduction, train_targets)},
                step=epoch,
            )
            wandb.log(
                {"val_median": self._median(val_predictions, val_targets)}, step=epoch
            )
            wandb.log(
                {"test_median": self._median(test_predictions, test_targets)},
                step=epoch,
            )

            wandb.log(
                {"train_rmse": self._rmse(training_preduction, train_targets)},
                step=epoch,
            )
            wandb.log(
                {"val_rmse": self._rmse(val_predictions, val_targets)}, step=epoch
            )
            wandb.log(
                {"test_rmse": self._rmse(test_predictions, test_targets)}, step=epoch
            )

            wandb.log(
                {
                    "train_r_squared": self._r_squared(
                        training_preduction, train_targets
                    )
                },
                step=epoch,
            )
            wandb.log(
                {"val_r_squared": self._r_squared(val_predictions, val_targets)},
                step=epoch,
            )
            wandb.log(
                {"test_r_squared": self._r_squared(test_predictions, test_targets)},
                step=epoch,
            )

        return self.model


class Trainer_fingerprint:
    def __init__(self, hyper_param):
        self.hyper_param = hyper_param
        self.model = hyper_param["model"]["method"]

    def fit(self, train_features, train_targets):
        if self.hyper_param["model"]["name"] == "LightGbm":
            self.fit_lightgbm(train_features, train_targets)
        elif self.hyper_param["model"]["name"] == "RandomForest":
            self.fit_randomforest(train_features, train_targets)
        elif self.hyper_param["model"]["name"] == "NN":
            self.trainer_nn = self.fit_nn(train_features, train_targets)
        else:
            raise NotImplementedError

    def fit_lightgbm(self, train_features, train_targets):
        train_dataset = lgbm.Dataset(train_features, label=train_targets)

        self.model = lgbm.train(
            params=self.hyper_param["model"]["parameters"],
            train_set=train_dataset,
            num_boost_round=10000,
        )

    def fit_randomforest(self, train_features, train_targets):
        self.model = self.model(**self.hyper_param["model"]["parameters"])
        self.model.fit(train_features, train_targets)

    def fit_nn(self, train_features, train_targets):
        trainer = Trainer_NN(self.hyper_param)

        # Create validation and training sets.
        dataset = MolecularDataset(train_features, train_targets)

        # divide into subsets
        train_set, valid_set = torch.utils.data.random_split(
            dataset,
            [
                len(dataset)
                - round(self.hyper_param["data"]["val_set_size"] * len(dataset)),
                round(self.hyper_param["data"]["val_set_size"] * len(dataset)),
            ],
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=self.hyper_param["batch_size"],
            shuffle=False,
            pin_memory=True,
        )
        validation_loader = torch.utils.data.DataLoader(
            dataset=valid_set,
            batch_size=self.hyper_param["batch_size"],
            shuffle=False,
            pin_memory=True,
        )
        # run
        print(train_loader)
        print(validation_loader)
        model = trainer.run(
            train_loader,
            validation_loader,
            n_epochs=self.hyper_param["n_epochs"],
        )
        self.model = model
        return trainer


class Trainer_NN:
    def __init__(self, hyper_param):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Using device " + str(self.device) + ".")
        # set up model
        model = hyper_param["model"]["method"](**hyper_param["model"]["parameters"])
        # set up optimizer and scheduler
        optimizer = hyper_param["optimizer"]["method"](
            model.parameters(), **hyper_param["optimizer"]["parameters"]
        )
        scheduler = hyper_param["scheduler"]["method"](
            optimizer, **hyper_param["scheduler"]["parameters"]
        )
        self._model = model.to(self.device)
        self._optimizer = optimizer
        self._scheduler = scheduler

        self._gradient_accumulation_splits = 1

    @property
    def model(self):
        """Getter for model."""
        return self._model

    def _train(self, train_loader):
        """Performs a full training step. Depending on the setting for gradient
        accumulation, performs backward pass only every n batch.

        Returns:
            float: The obtained training loss.
        """
        self._model.train()
        loss_all = 0

        for batch_idx, (features, targets) in enumerate(train_loader):
            batch = features.to(self.device)
            loss = F.mse_loss(self._model(batch), targets.to(self.device))
            loss.backward()
            loss_all += loss.item()

            # gradient accumulation
            if ((batch_idx + 1) % self._gradient_accumulation_splits == 0) or (
                batch_idx + 1 == len(train_loader)
            ):
                self._optimizer.step()
                self._optimizer.zero_grad()

        return loss_all / len(train_loader.dataset)

    def _mae(self, predictions, targets):
        errors = np.abs(predictions - targets)
        return np.mean(errors)

    def _median(self, predictions, targets):
        errors = np.abs(predictions - targets)
        return np.median(errors)

    def _rmse(self, predictions, targets):
        errors = np.abs(predictions - targets)
        return np.sqrt(np.mean(np.power(errors, 2)))

    def _r_squared(self, predictions, targets):
        target_mean = np.mean(targets)
        return 1 - (
            np.sum(np.power(targets - predictions, 2))
            / np.sum(np.power(targets - target_mean, 2))
        )

    def predict_batch(
        self, batch, target_means=0, target_stds=1, target_offset_dict=None
    ):
        """Makes predictions on a given batch.

        Returns:
            list: The predictions.
        """

        self._model.eval()
        batch = batch.to(self.device)

        # get predictions for batch
        predictions = (self.model(batch).cpu().detach().numpy()).tolist()

        return predictions

    def predict(self, features):
        # Create validation and training sets.
        dataset = MolecularDataset(features, [0] * len(features))
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=8,
            shuffle=False,
            pin_memory=True,
        )
        return self.predict_loader(loader)

    def predict_loader(
        self, loader, target_means=0, target_stds=1, target_offset_dict=None
    ):
        """Makes predictions on a given dataloader.

        Returns:
            list: The predictions.
        """

        predictions = []

        for features, targets in loader:
            predictions.extend(
                self.predict_batch(
                    features,
                    target_means=target_means,
                    target_stds=target_stds,
                    target_offset_dict=target_offset_dict,
                )
            )
        return predictions

    def run(self, train_loader, val_loader, n_epochs):
        """Runs a full training loop with automatic metric logging through
        wandb.

        Args:
            train_loader (Dataloader): The dataloader for the training points.
            val_loader (Dataloader): The dataloader for the validation points.

        Returns:
            model: The trained model.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader

        # get targets off all sets
        train_targets = []
        for features, targets in train_loader:
            train_targets.extend(targets)
        val_targets = []
        for features, targets in val_loader:
            val_targets.extend(targets)

        for epoch in range(1, n_epochs + 1):
            # get learning rate from scheduler
            if self._scheduler is not None:
                lr = self._scheduler.optimizer.param_groups[0]["lr"]

            # training step
            loss = self._train(train_loader)

            # get predictions for all sets
            training_preduction = np.array(self.predict_loader(train_loader))
            val_predictions = np.array(
                self.predict_loader(
                    val_loader,
                )
            )

            # Calculating error
            train_error = self._mae(train_targets, training_preduction)
            val_error = self._mae(val_targets, val_predictions)

            # learning rate scheduler step
            if self._scheduler is not None:
                self._scheduler.step(val_error)

            output_line = (
                f"Epoch: {epoch:03d}, LR: {lr:7f}, Loss: {loss:.7f}, "
                f"Train MAE: {train_error:.7f}, "
                f"Val MAE: {val_error:.7f}"
            )
            print(output_line)

            # wandb logging
            wandb.log({"loss": loss}, step=epoch)
            wandb.log({"train_error": train_error}, step=epoch)
            wandb.log({"val_error": val_error}, step=epoch)

            wandb.log(
                {"train_mae": self._mae(training_preduction, train_targets)}, step=epoch
            )
            wandb.log({"val_mae": self._mae(val_predictions, val_targets)}, step=epoch)

            wandb.log(
                {"train_median": self._median(training_preduction, train_targets)},
                step=epoch,
            )
            wandb.log(
                {"val_median": self._median(val_predictions, val_targets)}, step=epoch
            )

            wandb.log(
                {"train_rmse": self._rmse(training_preduction, train_targets)},
                step=epoch,
            )
            wandb.log(
                {"val_rmse": self._rmse(val_predictions, val_targets)}, step=epoch
            )
            wandb.log(
                {
                    "train_r_squared": self._r_squared(
                        training_preduction, train_targets
                    )
                },
                step=epoch,
            )
            wandb.log(
                {"val_r_squared": self._r_squared(val_predictions, val_targets)},
                step=epoch,
            )
        return self.model


class HyperParameterGuider:
    def __init__(self, hyper_param):
        self.hyper_param = hyper_param
        print(hyper_param)
        self.model = hyper_param["model"]["method"]

    def optimize_hyperparameters(self, training_features, training_targets):
        print(self.model)
        if "LightGbm" in self.hyper_param["model"]["name"]:
            self.tuned_model = self.optimize_lightgbm_parameters(
                training_features, training_targets
            )
            self.tuned_model.save_model(
                "final_best_model.txt",
                num_iteration=self.tuned_model.best_iteration,
            )
            best_params = self.tuned_model.params
            with open("lgbm_best_params.txt", "w") as f:
                f.write(json.dumps(best_params))

            wandb.save("final_best_model.txt")
            wandb.save("lgbm_best_params.txt")

        elif "RandomForest" in self.hyper_param["model"]["name"]:
            model = self.optimize_randomforest_parameters(
                training_features, training_targets
            )
            self.tuned_model = model
            with open("random_forrest_best_params.txt", "w") as f:
                f.write(json.dumps(self.tuned_model.best_params_))
            wandb.save("random_forrest_best_params.txt")

        return self.tuned_model

    def optimize_randomforest_parameters(self, training_features, training_targets):
        # Define the hyperparameter grid to search
        param_grid = {
            "n_estimators": [10, 200, 300],
            "min_samples_leaf": [2, 3, 4],
        }

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            self.model(),
            param_grid,
            cv=self.hyper_param["cv"],
            refit=True,
            verbose=4,
            n_jobs=self.hyper_param["njobs"],
        )
        model = grid_search.fit(training_features, training_targets)
        return model

    def optimize_lightgbm_parameters(self, training_features, training_targets):
        model = optimize_lgbm(training_features, training_targets)
        return model

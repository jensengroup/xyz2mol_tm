import random

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader


class MolecularDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def set_global_seed(seed: int) -> None:
    """Sets the random seed for python, numpy and pytorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_feature_matrix_dict(dataset, feature_keys: list[str]):
    """Gets a dictionary of feature matrices based on a given list of feature
    keys.

    Args:
        dataset (Data): The pytorch dataset.
        feature_keys (list[str]): The list of feature keys to get feature matrices for.

    Returns:
        dict: The dict of feature matrices.
    """
    feature_matrix_dict = {}

    for data in dataset:
        for feature_key in feature_keys:
            if feature_key not in feature_matrix_dict.keys():
                feature_matrix_dict[feature_key] = []

            if type(data[feature_key].detach().numpy()[0]) == np.ndarray:
                feature_matrix_dict[feature_key].extend(
                    data[feature_key].detach().numpy()
                )
            else:
                feature_matrix_dict[feature_key].append(
                    data[feature_key].detach().numpy()
                )

    for feature_key in feature_matrix_dict.keys():
        feature_matrix_dict[feature_key] = np.array(feature_matrix_dict[feature_key])

    return feature_matrix_dict


def standard_scale_dataset(dataset, feature_matrix_dict: dict):
    """Standard scales a dataset based on a given dictionary of
    feature_matrices. Each feature that is present in the feature matrix dict
    will be scaled in the dataset.

    Args:
        dataset (Data): The pytorch dataset.
        feature_matrix_dict (dict): The feature matrix dict where each key should correspond to a feature in the data.

    Returns:
        Data: The scaled pytorch dataset.
    """
    feature_means = {}
    feature_stds = {}

    for feature_key in feature_matrix_dict.keys():
        feature_means[feature_key] = np.mean(feature_matrix_dict[feature_key], axis=0)
        feature_stds[feature_key] = np.std(feature_matrix_dict[feature_key], axis=0)

        # replace zero stds with ones
        feature_stds[feature_key][feature_stds[feature_key] == 0] = 1

    for data in dataset:
        for feature_key in feature_matrix_dict.keys():
            data[feature_key] = (
                data[feature_key] - feature_means[feature_key]
            ) / feature_stds[feature_key]

    return dataset


def get_feature_means_from_feature_matrix_dict(
    feature_matrix_dict: dict, feature_key: str
):
    return np.mean(feature_matrix_dict[feature_key], axis=0, keepdims=False)


def get_feature_stds_from_feature_matrix_dict(
    feature_matrix_dict: dict, feature_key: str
):
    return np.std(feature_matrix_dict[feature_key], axis=0, keepdims=False)


def calculate_r_squared(predictions, targets):
    """Calculates the R^2 value for given y and y_hat.

    Returns:
        float: The R^2 value.
    """
    target_mean = np.mean(targets)
    return 1 - (
        np.sum(np.power(targets - predictions, 2))
        / np.sum(np.power(targets - target_mean, 2))
    )


def get_target_list(
    loader: DataLoader, target_means=[0], target_stds=[1], target_offset_dict=None
):
    """Gets the list of targets of a dataloader.

    Args:
        loader (Dataloader): The pytorch_geometric dataloader.
        target_means (np.array): An array of target means.
        target_stds (np.array): An array of target stds.
        target_offset_dict (dict): A dictionary that contains ID - Offset pairs that specifies offsets to be added for each individual
            data point.

    Returns:
        list: The list of targets.
    """

    targets = []
    for batch in loader:
        targets.extend(
            get_target_list_from_batch(
                batch,
                target_means=target_means,
                target_stds=target_stds,
                target_offset_dict=target_offset_dict,
            )
        )

    return targets


def get_target_list_from_batch(
    batch: Batch, target_means=[0], target_stds=[1], target_offset_dict=None
):
    """Gets the list of targets of a batch.

    Args:
        batch (Batch): The pytorch_geometric batch.
        target_means (np.array): An array of target means.
        target_stds (np.array): An array of target stds.
        target_offset_dict (dict): A dictionary that contains ID - Offset pairs that specifies offsets to be added for each individual
            data point.

    Returns:
        list: The list of targets.
    """

    offset = 0
    if target_offset_dict is not None:
        offset = np.array([target_offset_dict[i] for i in batch.id])

    targets = (batch.y.numpy() * target_stds + target_means + offset).tolist()

    return targets


class Tools:
    @staticmethod
    def get_one_hot_encoded_feature_dict(
        feature_dict: dict, class_feature_dict: dict
    ) -> dict:
        """Gets the one-hot encoding of a given feature list according to a
        given dict.

        Returns:
            dict: Dict of features.
        """

        one_hot_encoded_feature_dict = {}

        for key in feature_dict.keys():
            if key in class_feature_dict.keys():
                one_hot_encoded_feature_dict[key] = Tools.get_one_hot_encoding(
                    len(class_feature_dict[key]),
                    class_feature_dict[key].index(feature_dict[key]),
                )
            else:
                one_hot_encoded_feature_dict[key] = feature_dict[key]

        return one_hot_encoded_feature_dict

    @staticmethod
    def get_one_hot_encoded_feature_list(feature_dict: dict, class_feature_dict: dict):
        """Gets the one-hot encoding of a given feature list according to a
        given dict.

        Returns:
            list[float]: The one-hot encoded feature list.
        """
        one_hot_encoded_feature_dict = Tools.get_one_hot_encoded_feature_dict(
            feature_dict, class_feature_dict
        )

        return Tools.flatten_list(
            [
                one_hot_encoded_feature_dict[key]
                for key in one_hot_encoded_feature_dict.keys()
            ]
        )

    @staticmethod
    def get_one_hot_encoding(n_classes: int, class_number: int):
        """Helper function that the one hot encoding for one specific element
        by specifying the number of classes and the class of the current
        element.

        Raises:
            ValueError: If a class number is requested that is higher than the maximum number of classes.

        Returns:
            list[int]: The one hot encoding of one element.
        """
        if class_number >= n_classes:
            raise ValueError(
                "Cannot get one hot encoding for a class number higher than the number of classes."
            )

        # return empty list if there is only one type
        if n_classes == 1:
            return []

        return [1 if x == class_number else 0 for x in list(range(n_classes))]

    @staticmethod
    def get_class_feature_keys(feature_dict):
        """Takes a feature dict a determines at which keys non-numerical class
        features are used.

        Returns:
            list[str]: A list with keys which correspond to non-numerical class features.
        """
        class_feature_keys = []

        # get indices of features that are not numeric and need to be one-hot encoded
        for key in feature_dict.keys():
            if (
                not type(feature_dict[key]) == int
                and not type(feature_dict[key]) == float
            ):
                class_feature_keys.append(key)

        return class_feature_keys

    @staticmethod
    def get_class_feature_indices(feature_list):
        """Takes a feature list a determines at which positions non-numerical
        class features are used.

        Returns:
            list[int]: A list with indices denoting at which positions non-numerical class features are used.
        """
        class_feature_indices = []

        # get indices of features that are not numeric and need to be one-hot encoded
        for i in range(len(feature_list)):
            if not type(feature_list[i]) == int and not type(feature_list[i]) == float:
                class_feature_indices.append(i)

        return class_feature_indices

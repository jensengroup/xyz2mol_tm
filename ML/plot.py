import matplotlib.pyplot as plt
import numpy as np
import wandb
from scipy.interpolate import interpn
from tools import calculate_r_squared


def plot_metal_center_group_histogram(
    train_dataset,
    val_dataset,
    test_dataset,
    meta_data_dict: dict,
    file_path="./image.png",
):
    train_group_counts = np.zeros(10)
    for graph in train_dataset:
        train_group_counts[meta_data_dict[graph.id]["metal_center_group"] - 3] += 1
    train_group_counts = train_group_counts / len(train_dataset)

    val_group_counts = np.zeros(10)
    for graph in val_dataset:
        val_group_counts[meta_data_dict[graph.id]["metal_center_group"] - 3] += 1
    val_group_counts = val_group_counts / len(val_dataset)

    test_group_counts = np.zeros(10)
    for graph in test_dataset:
        test_group_counts[meta_data_dict[graph.id]["metal_center_group"] - 3] += 1
    test_group_counts = test_group_counts / len(test_dataset)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(3, 13) - 0.25, train_group_counts, width=0.35, label="Train")
    ax.bar(np.arange(3, 13), val_group_counts, width=0.35, label="Val")
    ax.bar(np.arange(3, 13) + 0.25, test_group_counts, width=0.35, label="Test")
    ax.set_xlabel("Metal center group")
    ax.set_ylabel("Relative size")
    ax.set_xticks([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ax.legend()

    fig.savefig(file_path, format="png", dpi=300, bbox_inches="tight")


def plot_target_histogram(train_true_values, test_true_values, file_path="./image.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(train_true_values, label="Train", alpha=0.5)
    ax.hist(
        test_true_values,
        label="Test",
        alpha=0.5,
    )
    ax.set_xlabel("Target value")
    ax.set_ylabel("Frequency")
    ax.legend()

    fig.savefig(file_path, format="png", dpi=300, bbox_inches="tight")


def plot_correlation(
    predicted_values: list, true_values: list, file_path="./image.png", target_unit=None
):
    # cast lists to arrays
    predicted_values = np.array(predicted_values)
    true_values = np.array(true_values)

    # set up canvas
    fig, ax = plt.subplots(figsize=(5, 5))

    # taken from https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib/53865762#53865762
    # get interpolated point densities
    data, x_e, y_e = np.histogram2d(
        predicted_values, true_values, bins=20, density=True
    )
    z = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([predicted_values, true_values]).T,
        method="splinef2d",
        bounds_error=False,
    )

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    predicted_values, true_values, z = predicted_values[idx], true_values[idx], z[idx]

    # set base points with density coloring
    ax.scatter(true_values, predicted_values, c=z, cmap="Purples")

    # get min and max values
    min_value = min(predicted_values)
    max_value = max(predicted_values)

    # regression line
    z = np.polyfit(predicted_values, true_values, 1)
    p = np.poly1d(z)
    ax.plot([min_value, max_value], [p(min_value), p(max_value)], "k--")

    # formatting
    ax.text(
        0.2,
        0.9,
        "R² = "
        + str(
            np.round(
                calculate_r_squared(np.array(predicted_values), np.array(true_values)),
                decimals=3,
            )
        ),
        size=10,
        color="Purple",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_xlabel(f"True values ({target_unit})")
    ax.set_ylabel(f"Predicted values ({target_unit})")

    # set same length axis ranges with 5% margin of max value
    ax.set_xlim([min_value - 0.05 * max_value, max_value + 0.05 * max_value])
    ax.set_ylim([min_value - 0.05 * max_value, max_value + 0.05 * max_value])

    fig.savefig(file_path, format="png", dpi=300, bbox_inches="tight")


def plot_error_by_metal_center_group(
    predicted_values: list,
    true_values: list,
    metal_center_groups: list,
    file_path="./image.png",
):
    group_counts = np.zeros(10)
    group_accumulated_errors = np.zeros(10)
    for i in range(len(predicted_values)):
        group_accumulated_errors[metal_center_groups[i] - 3] += np.abs(
            predicted_values[i] - true_values[i]
        )
        group_counts[metal_center_groups[i] - 3] += 1

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(3, 13), group_accumulated_errors / group_counts)
    ax.set_xlabel("Metal center group")
    ax.set_ylabel("Mean average deviation")

    fig.savefig(file_path, format="png", dpi=300, bbox_inches="tight")


def wandb_plot_error_by_metal_center_group(
    predicted_values: list, true_values: list, metal_center_groups: list
):
    group_counts = np.zeros(10)
    group_accumulated_errors = np.zeros(10)
    for i in range(len(predicted_values)):
        group_accumulated_errors[metal_center_groups[i] - 3] += np.abs(
            predicted_values[i] - true_values[i]
        )
        group_counts[metal_center_groups[i] - 3] += 1

    data = [
        [name, prec]
        for (name, prec) in zip(
            np.arange(3, 13), group_accumulated_errors / group_counts
        )
    ]
    table = wandb.Table(data=data, columns=["Metal center group", "Average error"])

    return wandb.plot.bar(
        table,
        "Metal center group",
        "Average error",
        title="Error by metal center group",
    )


if __name__ == "__main__":
    # Simulate plots
    val = np.random.uniform(low=0.5, high=13.3, size=(5000,))
    val_y = val + np.random.normal(0, 0.5, len(val))
    plot_correlation(
        val,
        val_y,
        target_unit="Test",
    )

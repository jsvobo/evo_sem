# Different datasets for the task
# small dimensional dataset (2D) inside unitary cube
# big dimensional dataset (50D) inside unitary cube
# one realistic dataset with categorical data


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_small_dataset(n_samples=500, n_features=2, random_seed=42):
    np.random.seed(random_seed)
    class1 = np.random.normal(loc=0, scale=0.3, size=(n_samples // 2, n_features))

    r = 1  # radius of the ring
    angles = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    radii = np.random.normal(loc=r, scale=0.2, size=n_samples // 2)
    class2_x = radii * np.cos(angles)
    class2_y = radii * np.sin(angles)
    class2 = np.column_stack((class2_x, class2_y))

    data = np.vstack((class1, class2))
    labels = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    data = 2 * (data - data_min) / (data_max - data_min) - 1

    data[0, :] = 0.9
    data[1, :] = 0.91

    feature_ranges = [(data[:, i].min(), data[:, i].max()) for i in range(n_features)]
    return {
        "data": pd.DataFrame(data, columns=[f"feature_{i}" for i in range(n_features)]),
        "labels": labels,
        "feature_bounds": feature_ranges,
    }


def generate_multidimensional_dataset(n_samples=1000, n_features=50, random_seed=42):
    np.random.seed(random_seed)
    class1 = np.random.normal(loc=0, scale=0.3, size=(n_samples // 2, n_features))

    r = 1  # radius of the ring
    angles = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    radii = np.random.normal(loc=r, scale=0.2, size=n_samples // 2)
    class2_x = radii * np.cos(angles)
    class2_y = radii * np.sin(angles)
    class2 = np.column_stack((class2_x, class2_y))

    data = np.vstack((class1, class2))
    labels = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    data = 2 * (data - data_min) / (data_max - data_min) - 1

    data[0, :] = 0.9
    data[1, :] = 0.91

    # Permute rows randomly
    permutation = np.random.permutation(n_samples)
    data = data[permutation]
    labels = labels[permutation]

    feature_ranges = [(data[:, i].min(), data[:, i].max()) for i in range(n_features)]
    return {
        "data": pd.DataFrame(data, columns=[f"feature_{i}" for i in range(n_features)]),
        "labels": labels,
        "feature_bounds": feature_ranges,
    }


def visualize_small_dataset(data, labels, filename="small_dataset.png"):
    plt.figure(figsize=(6, 6))
    first = data[labels == 0]
    plt.scatter(
        first["feature_0"], first["feature_1"], marker="x", color="r", s=70, alpha=0.9
    )

    second = data[labels == 1]
    plt.scatter(
        second["feature_0"], second["feature_1"], marker="o", color="b", alpha=0.6
    )

    plt.axhline(0, color="k", lw=0.7)
    plt.axvline(0, color="k", lw=0.7)
    plt.xticks([])
    plt.yticks([])

    plt.title("Small Dimensional Dataset Visualization")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()


if __name__ == "__main__":
    dataset = generate_small_dataset()
    visualize_small_dataset(dataset["data"], dataset["labels"])

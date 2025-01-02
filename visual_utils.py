import numpy as np
import matplotlib.pyplot as plt
from decision_tree import Tree


def visualise_tree_decision(tree: Tree, dataset: dict):
    plt.figure(figsize=(6, 6))

    bounds = dataset["feature_bounds"]

    if len(dataset["feature_bounds"]) == 2:
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = tree.inference(grid)
        Z = Z.reshape(xx.shape)

        plt.contourf(
            xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 10), cmap=plt.cm.Spectral_r
        )

        # Plot the dataset points
        class_0 = dataset["data"][dataset["labels"] == 0]
        class_1 = dataset["data"][dataset["labels"] == 1]
        plt.scatter(
            class_0["feature_0"],
            class_0["feature_1"],
            c="blue",
            label="Class 0",
            edgecolor="k",
        )
        plt.scatter(
            class_1["feature_0"],
            class_1["feature_1"],
            c="red",
            label="Class 1",
            edgecolor="k",
        )

        plt.xticks([])
        plt.yticks([])

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.title("Decision Boundary and Data Points")
        plt.show()

    else:
        print("Cannot visualize dataset with more than 2 dimensions.")


def simple_graph(array, xlabel, ylabel, title):
    plt.plot(range(len(array)), array)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

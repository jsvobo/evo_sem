from decision_tree import Tree
import numpy as np
import pandas as pd


def fitness(tree, dataset):
    return tree.calculate_accuracy(dataset)


class BaggingFitness:

    def __init__(self, dataset, num_bags=8, bagging_frac=1.0):
        np.random.seed(10)
        self.dataset = dataset
        self.bagging_frac = bagging_frac
        self.num_bags = num_bags
        data = dataset["data"].to_numpy()

        print("dataset keys:", dataset.keys())
        num_data = len(dataset["data"])
        print(f"Number of data: {num_data}")

        self.bag_size = int(num_data * self.bagging_frac)
        self.bags = []
        for i in range(self.num_bags):
            indices = np.random.choice(num_data, self.bag_size, replace=True)
            bag_dict = {
                "data": pd.DataFrame(data[indices]),
                "labels": dataset["labels"][indices],
                "feature_bounds": dataset["feature_bounds"],
            }
            self.bags.append(bag_dict)

    def fitness(self, tree):
        index = np.random.randint(0, self.num_bags)
        return tree.calculate_accuracy(self.bags[index])


def bagging_fitness(tree, bags):
    return bags.fitness(tree)

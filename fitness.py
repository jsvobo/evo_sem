from decision_tree import Tree


def fitness(tree, dataset):
    return tree.calculate_accuracy(dataset)


def fitness_random_subset(tree, subset_size, dataset):
    """
    This fitness function is used for the local search and algorithm. It takes a random subset of the dataset and calculates the accuracy of the tree on that subset.

    """
    size_wanted = dataset.shape[1] // subset_size
    random_indices = np.random.choice(dataset.shape[1], size_wanted, replace=False)
    subdataset = dataset[:, random_indices]
    return tree.calculate_accuracy(subdataset)

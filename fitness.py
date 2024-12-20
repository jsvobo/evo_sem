from decision_tree import Tree


def fitness(tree, dataset):
    return tree.calculate_accuracy(dataset)

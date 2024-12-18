import numpy as np


class Node:

    def __init__(self, depth, attribute=None, threshold=None, parity=None):
        # children of the node
        self.left = TerminalNode(0)
        self.right = TerminalNode(1)

        # decision
        self.attribute = attribute
        self.threshold = threshold
        self.parity = parity

        # tree info
        self.depth = depth

    def __str__(self):
        return (
            "f: "
            + str(self.attribute)
            + ": "
            + str(self.parity)
            + " thr: "
            + str(self.threshold)
        )

    def randomly_init(self, feature_bounds, depth, max_depth):
        feature = np.random.randint(0, len(feature_bounds))
        self.attribute = feature
        self.threshold = np.random.uniform(
            feature_bounds[feature][0], feature_bounds[feature][1]
        )
        self.parity = np.random.choice([-1, 1])

        if depth + 1 < max_depth:
            self.left = Node(
                depth=depth + 1, attribute=None, threshold=None, parity=None
            )
            self.left.randomly_init(feature_bounds, depth + 1, max_depth)

            self.right = Node(
                depth=depth + 1, attribute=None, threshold=None, parity=None
            )
            self.right.randomly_init(feature_bounds, depth + 1, max_depth)

    def infer(self, data, indices) -> np.ndarray:
        assert self.attribute is not None, "attribute is not set"
        # remember the indices of the data you pass into the subtrees
        # then infer the left and right, put list together
        attributes_extracted = data[indices, self.attribute]
        indices_under = np.where(attributes_extracted < self.threshold)[0]
        indices_over = np.where(attributes_extracted >= self.threshold)[0]

        if self.parity == 0:  # send what is under to the left
            left_indices = indices_under
            right_indices = indices_over
        else:  # self.parity=1, send what is over to the left+
            left_indices = indices_over
            right_indices = indices_under

        left_classes = self.right.infer(data, indices[left_indices])
        right_classes = self.left.infer(data, indices[right_indices])

        returned_classes = -np.ones(len(indices))
        returned_classes[left_indices] = left_classes
        returned_classes[right_indices] = right_classes

        return returned_classes


class TerminalNode:

    def __init__(self, value):
        assert value == 0 or value == 1, "invalid value passed to the terminal node"
        self.value = value

    def infer(self, data, indices):
        return np.full(len(indices), self.value)  # all is one class


class Tree:

    def __init__(self, feature_bounds, max_depth=4, random_seed=42):
        # random initialization of the tree. randomly create depth 2 tree
        np.random.seed(random_seed)
        self.root = Node(depth=0)
        self.root.randomly_init(feature_bounds, 0, max_depth)  # depth = 3
        self.features_bounds = feature_bounds

    def inference(self, data):
        return self.root.infer(data, np.arange(len(data)))

    def calculate_accuracy(self, dataset):
        data = dataset["data"].to_numpy()
        inferred_cl = self.inference(data)

        labels = dataset["labels"]
        accuracy = np.mean(inferred_cl == labels)
        return accuracy

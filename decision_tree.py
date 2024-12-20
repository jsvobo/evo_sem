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
            + "{:.3f}".format(self.threshold)
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

    def clever_init(self, feature_bounds, depth, max_depth, feature):
        len_features = len(feature_bounds)
        feature = (feature + 1) % len_features
        self.attribute = feature

        self.threshold = np.random.normal(
            loc=(feature_bounds[feature][0] + feature_bounds[feature][1]) / 2,
            scale=(feature_bounds[feature][1] - feature_bounds[feature][0]) / 6,
        )
        self.threshold = (
            self.threshold
            if self.threshold < feature_bounds[feature][1]
            else feature_bounds[feature][1]
        )
        self.threshold = (
            self.threshold
            if self.threshold > feature_bounds[feature][0]
            else feature_bounds[feature][0]
        )
        self.parity = np.random.choice([-1, 1])

        if depth + 1 < max_depth:
            self.left = Node(
                depth=depth + 1, attribute=None, threshold=None, parity=None
            )
            self.left.clever_init(feature_bounds, depth + 1, max_depth, feature)

            self.right = Node(
                depth=depth + 1, attribute=None, threshold=None, parity=None
            )
            self.right.clever_init(feature_bounds, depth + 1, max_depth, feature)

    def randomly_change_node(self, feature_bounds, depth, p_add=0.25):
        feature = np.random.randint(0, len(feature_bounds))
        self.attribute = feature
        self.threshold = np.random.uniform(
            feature_bounds[feature][0], feature_bounds[feature][1]
        )
        self.parity = np.random.choice([-1, 1])

        if np.random.rand() < p_add:
            self.left = Node(
                depth=depth + 1, attribute=None, threshold=None, parity=None
            )
            self.left.randomly_change_node(feature_bounds, depth + 1, p_add * 0.9)

            self.right = Node(
                depth=depth + 1, attribute=None, threshold=None, parity=None
            )
            self.right.randomly_change_node(feature_bounds, depth + 1, p_add * 0.9)

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

    def __str__(self):
        return "leaf: " + str(self.value)


class Tree:

    def __init__(
        self, feature_bounds, generation_type="basic", max_depth=3, p_add=0.25
    ):
        # random initialization of the tree. randomly create depth 2 tree
        self.root = Node(depth=0)
        self.features_bounds = feature_bounds

        if generation_type == "basic":
            self.basic_generation(max_depth)
        elif generation_type == "clever":
            self.clever_generation(max_depth)
        else:
            self.coinflip_generation(p_add)

        self.root.randomly_init(feature_bounds, 0, max_depth)  # depth = 3

    def inference(self, data):
        return self.root.infer(data, np.arange(len(data)))

    def calculate_accuracy(self, dataset):
        data = dataset["data"].to_numpy()
        inferred_cl = self.inference(data)

        labels = dataset["labels"]
        accuracy = np.mean(inferred_cl == labels)
        return accuracy

    def basic_generation(self, max_depth=4):
        self.root.randomly_init(self.features_bounds, depth=0, max_depth=max_depth)

    def coinflip_generation(self, p_add):
        self.root.randomly_change_node(self.features_bounds, depth=0, p_add=p_add)

    def clever_generation(self, max_depth=4):
        start_feature = np.random.randint(0, len(self.features_bounds))
        self.root.clever_init(
            self.features_bounds, depth=0, max_depth=max_depth, feature=start_feature
        )

    def _traverse(self, node):
        this_string = str(node)
        if isinstance(node, Node):
            left = str(node.left)
            right = str(node.right)
            print(this_string + " left: " + left + " right: " + right)

            self._traverse(node.left)
            self._traverse(node.right)

    def print_tree_traverse(self):
        self._traverse(self.root)

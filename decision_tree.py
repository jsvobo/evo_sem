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
            + " depth: "
            + str(self.depth)
        )

    def copy(self):
        new_node = Node(
            depth=self.depth,
            attribute=self.attribute,
            threshold=self.threshold,
            parity=self.parity,
        )

        if not self.left.is_terminal():
            new_node.left = self.left.copy()
        if not self.right.is_terminal():
            new_node.right = self.right.copy()

        return new_node

    def randomly_init(self, feature_bounds, depth, max_depth):
        feature = np.random.randint(0, len(feature_bounds))
        self.attribute = feature
        self.threshold = np.random.uniform(
            feature_bounds[feature][0], feature_bounds[feature][1]
        )
        self.parity = np.random.choice([-1, 1])

        if depth + 1 <= max_depth:
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

        if depth + 1 <= max_depth:
            self.left = Node(
                depth=depth + 1, attribute=None, threshold=None, parity=None
            )
            self.left.clever_init(feature_bounds, depth + 1, max_depth, feature)

            self.right = Node(
                depth=depth + 1, attribute=None, threshold=None, parity=None
            )
            self.right.clever_init(feature_bounds, depth + 1, max_depth, feature)

    def coinflip_init(self, feature_bounds, depth, p_add):
        feature = np.random.randint(0, len(feature_bounds))
        self.attribute = feature
        self.threshold = np.random.uniform(
            feature_bounds[feature][0], feature_bounds[feature][1]
        )
        self.parity = np.random.choice([-1, 1])

        prog = np.random.rand()
        # print("progs?", prog, " against: ", p_add, " depth: ", depth)
        if prog <= p_add:
            self.left = Node(
                depth=depth + 1, attribute=None, threshold=None, parity=None
            )
            self.left.coinflip_init(feature_bounds, depth + 1, p_add * 0.9)

            self.right = Node(
                depth=depth + 1, attribute=None, threshold=None, parity=None
            )
            self.right.coinflip_init(feature_bounds, depth + 1, p_add * 0.9)

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

    def is_terminal(self):
        return False

    def get_subtree_depth(self):
        return max(
            self.depth, self.left.get_subtree_depth(), self.right.get_subtree_depth()
        )

    def get_subtree_size(self):
        return 1 + self.left.get_subtree_size() + self.right.get_subtree_size()


class TerminalNode:

    def __init__(self, value):
        assert value == 0 or value == 1, "invalid value passed to the terminal node"
        self.value = value

    def infer(self, data, indices):
        return np.full(len(indices), self.value)  # all is one class

    def __str__(self):
        return "leaf: " + str(self.value)

    def is_terminal(self):
        return True

    def get_subtree_depth(self):
        return 0

    def get_subtree_size(self):
        return 0


class Tree:

    def __init__(
        self, feature_bounds, generation_type="basic", max_depth=3, p_add=0.70
    ):
        # random initialization of the tree. randomly create depth 2 tree
        self.root = Node(depth=0)
        self.features_bounds = feature_bounds

        if generation_type == "basic":
            self.basic_generation(max_depth)
        elif generation_type == "clever":
            self.clever_generation(max_depth)
        elif generation_type == "coinflip":
            self.coinflip_generation(p_add)
        else:
            raise ValueError("invalid generation type")

    def inference(self, data):
        return self.root.infer(data, np.arange(len(data)))

    def calculate_accuracy(self, dataset):
        data = dataset["data"].to_numpy()
        inferred_cl = self.inference(data)

        labels = dataset["labels"]
        accuracy = np.mean(inferred_cl == labels)
        return accuracy

    def basic_generation(self, max_depth):
        assert max_depth > 0, "invalid max_depth"
        self.root.randomly_init(self.features_bounds, depth=0, max_depth=max_depth)

    def coinflip_generation(self, p_add):
        assert 0 <= p_add <= 1, "invalid p_add"
        self.root.coinflip_init(self.features_bounds, depth=0, p_add=p_add)

    def clever_generation(self, max_depth):
        assert max_depth > 0, "invalid max_depth"
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

    def copy(self):
        new_tree = Tree(self.features_bounds)
        new_tree.root = self.root.copy()
        return new_tree

    def depth(self):
        return self.root.get_subtree_depth()

    def size(self):
        return self.root.get_subtree_size()

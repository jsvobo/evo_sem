from decision_tree import Tree, Node, TerminalNode
import numpy as np


def _iterate_nodes(node, prob, function):
    # try to prog, AND iterate further!

    if np.random.rand() < prob:
        # print("Perturbing node: ", str(node))
        node = function(node)

    if not node.left.is_terminal():
        node.left = _iterate_nodes(node.left, prob, function)
    if not node.right.is_terminal():
        node.right = _iterate_nodes(node.right, prob, function)
    return node


def _iterate_leaves(node, prob, function):
    # apply if terminal and prob progs
    if node.left.is_terminal():  # if terminal one above is changed?
        if np.random.rand() < prob:
            node = function(node)
        return node  # maybe unchanged?

    # iterate not teminal nodes
    node.left = _iterate_leaves(node.left, prob, function)
    node.right = _iterate_leaves(node.right, prob, function)
    return node


def perturb_change_value_normal(tree, prob_change, feature_bounds):
    sigma = 0.1

    def change_value(node):
        change = np.random.normal(loc=0, scale=sigma)
        node.threshold += change
        node.threshold = np.clip(
            node.threshold,
            feature_bounds[node.attribute][0],
            feature_bounds[node.attribute][1],
        )
        return node

    _iterate_nodes(tree.root, prob_change, change_value)


def perturb_randomly_add_to_leaf(tree, prob_add, feature_bounds):
    # go down, find some random laf add few random rows to it.
    def grow_leaf(node):
        new_node = Node(depth=node.depth + 1)
        new_node.randomly_init(
            feature_bounds, node.depth + 1, node.depth + 2
        )  # add a row to the leaf
        return new_node

    _iterate_leaves(tree.root, prob_add, grow_leaf)


def perturb_randomly_prune(tree, prob_prune):

    def prune(node):
        if node.depth <= 3:
            return node

        node.left = TerminalNode(value=0)
        node.right = TerminalNode(value=1)
        return node

    _iterate_nodes(tree.root, prob_prune, prune)


def randomly_grow(tree, feature_bounds, prob_add):
    tree_copy = tree.copy()
    perturb_randomly_add_to_leaf(tree_copy, prob_add, feature_bounds)
    return tree_copy


def combined_perturb(tree, feature_bounds, prob_value, prob_add, prob_prune):
    tree = tree.copy()

    perturb_randomly_prune(tree, prob_prune)
    perturb_change_value_normal(tree, prob_value, feature_bounds)
    perturb_randomly_add_to_leaf(tree, prob_add, feature_bounds)

    return tree

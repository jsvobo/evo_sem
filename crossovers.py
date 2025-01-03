import random


def get_node_and_parent_by_index(node, index, parent=None):
    stack = [(node, parent, None)]
    count = 0

    while stack:
        current, parent, side = stack.pop(0)
        if count == index:
            return current, parent, side

        count += 1
        if not current.left.is_terminal():
            stack.append((current.left, current, "left"))
        if not current.right.is_terminal():
            stack.append((current.right, current, "right"))

    return None, None, None


def crossover_swap_random_subtrees(parent1, parent2):

    size1 = parent1.size()
    size2 = parent2.size()

    child1 = parent1.copy()
    child2 = parent2.copy()

    if size1 <= 1 or size2 <= 1:
        return child1, child2

    # draw random node from children
    index1 = random.randint(1, size1 - 1)  # dont take root
    index2 = random.randint(1, size2 - 1)

    node1, node1_parent, side1 = get_node_and_parent_by_index(child1.root, index1)
    node2, node2_parent, side2 = get_node_and_parent_by_index(child2.root, index2)

    assert node1 is not None and node2 is not None, "Indexed node search returned None"

    # splice the subtrees together
    if side1 == "left":  # node 1 was taken from the left side
        node1_parent.left = node2
    else:
        node1_parent.right = node2

    if side2 == "left":  # node 2 was taken from the left side
        node2_parent.left = node1
    else:
        node2_parent.right = node1

    return child1, child2

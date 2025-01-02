import numpy as np
from decision_tree import Tree


def ls_first_improvement(
    fitness_fn,
    initialisation_fn,
    perturbation_fn,
    stop_cond=5000,
    one_step_max=500,
    feature_bounds=[],
):
    """
    Local search alg. using a given fitness function, initialisation function, perturbation function and distance matrix.
    The algorithm will run for a maximum number of steps or evaluations, whichever comes first.
    The algorithm will return the best solution found so far, along with its fitness value and the history of all solutions evaluated.
    """

    # where we store partial results from each iter.
    iterated_solutions = []

    # initialisation
    current_solution = Tree(
        feature_bounds, generation_type=initialisation_fn, max_depth=4, p_add=0.5
    )  # curried for the specific size beforehand!
    current_fitness = fitness_fn(current_solution)
    iteration = 0
    evals = 0

    overall_best = current_solution
    overall_best_fitness = current_fitness
    local_evals = 0
    iteration = 0

    # try to move, based on some perturbation function. is this better? if so, move there, if not, try again
    while True:
        if evals > stop_cond:
            break
        evals += 1
        local_evals += 1

        # perturb the current solution
        candidate_solution = perturbation_fn(current_solution)
        # calculate and compare fitness
        candidate_fitness = fitness_fn(candidate_solution)
        # print(current_solution.depth(), candidate_solution.depth())
        # print(current_fitness, candidate_fitness)

        # update the overall best solution
        if candidate_fitness > overall_best_fitness:
            overall_best = candidate_solution
            overall_best_fitness = candidate_fitness

        # store and go to the next iteration, even when going to the worse
        if candidate_fitness > current_fitness or local_evals >= one_step_max:
            current_solution = candidate_solution
            # print("Iteration: ", iteration)
            # current_solution.print_tree_traverse()
            current_fitness = candidate_fitness
            iterated_solutions.append(
                {
                    "iteration": iteration,
                    "fitness": current_fitness,
                    "solution": current_solution,
                    "local_evals": local_evals,
                }
            )

            iteration += 1
            local_evals = 0

    best_solution_globally = None
    best_fitness_globally = -np.inf
    for solution in iterated_solutions:
        if solution["fitness"] > best_fitness_globally:
            best_solution_globally = solution["solution"]
            best_fitness_globally = solution["fitness"]
    return {
        "history": iterated_solutions,
        "best_tree": best_solution_globally,
        "best_fitness": best_fitness_globally,
    }

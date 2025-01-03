import numpy as np


def combined_replacement(old, new, num_survivors, fitness_list_old, fitness_list_new):
    """
    Replace old population with new population
    :param old: list of individuals
    :param new: list of individuals
    :param num_survivors: number of survivors
    :param fitness_fn: fitness function
    :return: list of individuals
    """
    population_size = len(old)
    # combine old and new populations
    population = old + new
    fitness_list = np.concatenate((fitness_list_old, fitness_list_new))

    # sort population by fitness
    fitness_list = np.array(fitness_list)
    idcs = np.argsort(fitness_list)[::-1]
    # select survivors as ones with the best fitness
    smaller_list = idcs[:num_survivors]
    return [population[i] for i in smaller_list], fitness_list[smaller_list]


def tournament_selection(population, fitnesses, wanted_parents, tournament_size):
    selected = []
    total_size = len(population)
    for i in range(wanted_parents):  # sample new amount of parents??
        idcs = np.random.randint(low=0, high=total_size, size=tournament_size)
        best_idx = np.argmax(fitnesses[idcs])
        selected.append(population[best_idx])

    return selected  # population


def initialisation(init_fn, generation_size, fitness_fn):
    """
    Create initial population
    :param init_fn: function to create an individual
    :param generation_size: number of individuals in the population
    :return: list of individuals
    """
    instances = [init_fn() for _ in range(generation_size)]
    fitnesses = [fitness_fn(ind) for ind in instances]
    # calculate fitnesses stored inside all solutions (with violations)
    return instances, fitnesses


def apply_crossover(parent_list, operation, crossover_p):
    """
    Apply crossover to parents
    :param parents: list of individuals
    :param num_children: number of children to produce
    :return: list of children
    """
    children = []
    for i in range(len(parent_list) // 2):
        parent_a = parent_list[2 * i]
        parent_b = parent_list[2 * i + 1]
        if np.random.rand() < crossover_p:
            fabricated_children = operation(parent_a, parent_b)
            children.extend(fabricated_children)
        else:
            children.extend([parent_a, parent_b])
    return children


def evolutionary_algorithm(
    fitness_function,
    init_fn,
    crossover_fn,
    perturb_fn,
    prob_crossover=0.7,
    prob_mutation=0.1,
    population_size=100,
    max_evaluations=10000,
    tournament_size=20,
):

    # bookkepping
    results = []
    generation = 0
    wanted_parents = population_size

    best_overall_individual = None
    best_overall_fitness = -np.inf

    # Initialize variables
    population, fitness = initialisation(init_fn, population_size, fitness_function)
    evaluations = len(population)

    while evaluations < max_evaluations:
        # Calculate remaining evaluations and adjust offspring count
        remaining_evaluations = max_evaluations - evaluations
        offspring_count = min(remaining_evaluations, wanted_parents)

        fitness = np.array(fitness)

        # Selection
        parents = tournament_selection(
            population,
            fitness,
            offspring_count,
            tournament_size,
        )

        # Generate offsprings
        offspring = apply_crossover(parents, crossover_fn, prob_crossover)

        # Apply mutation to all offspring (based on p)
        offspring = [
            (perturb_fn(child) if np.random.rand() < prob_mutation else child)
            for child in offspring
        ]

        # Evaluate offspring fitness
        child_fitness_list = [
            fitness_function(i) for i in offspring
        ]  # calculate fitness inside offsprings
        evaluations += len(offspring)

        # Combine population and offspring, then select the best
        population, fitness = combined_replacement(
            population, offspring, population_size, fitness, child_fitness_list
        )

        # is better than previous?
        best_idx = np.argmax(fitness)
        local_best = population[best_idx]
        local_fitness = fitness[best_idx]

        # best overall solution
        if local_fitness > best_overall_fitness:
            best_overall_individual = local_best
            best_overall_fitness = local_fitness

        results.append(
            {
                "generation": generation,
                "avg_children_fitness": np.mean(child_fitness_list),
                "best_individual": local_best,
                "best_fitness": local_fitness,
                "evals": evaluations,
                "best_overall_individual": best_overall_individual,
                "best_overall_fitness": best_overall_fitness,
            }
        )

        generation += 1

    return {
        "history": results,
        "best_tree": best_overall_individual,
        "best_fitness": best_overall_fitness,
    }

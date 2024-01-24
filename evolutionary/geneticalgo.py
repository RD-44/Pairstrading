# This is an example of how the genetic algorithm can be used to solve a simple problem
# Find integers a, b, c, d in {0...30} such that a + 2b + 3c + 4d = 30

import numpy as np

cross_over_rate = 0.5
mutation_rate = 0.1
population_size = 10
chromosome_size = 4
num_generations = 100

def error(arr):
    return abs(arr[0] + 2 * arr[1] + 3 * arr[2] + 4 * arr[3] - 30)

def fitness(arr):
    return 1.0 / (1.0 + error(arr))

def generate_population():
    return np.random.randint(0, 31, size=(population_size, chromosome_size))

class Roulette:  # Used to create new population using a distribution weighted towards the fittest
    def __init__(self, population) -> None:
        fitness_scores = np.array([fitness(chromosome)
                                  for chromosome in population])
        self.probabilities = fitness_scores / np.sum(fitness_scores)
        self.population = population
        print("Score: ", np.sum(fitness_scores))

    def spin(self):
        rng = np.random.default_rng()
        # We sample from population using weighted distribution, with replacement
        return rng.choice(self.population, population_size, p=self.probabilities)


def successor(population):  # Gives the population one generation ahead of the one given
    roulette = Roulette(population)

    new_population = roulette.spin()

    cross_over = np.random.rand(population_size) < cross_over_rate

    temp = [np.copy(arr) for arr in new_population]

    parents = [i for i in range(len(cross_over)) if cross_over[i]]

    first = parents[-1] if len(parents) > 0 else None
    while len(parents) > 1:
        ind1, ind2 = parents.pop(), parents.pop()
        parent1, parent2 = temp[ind1], temp[ind2]

        cross_over_point = np.random.randint(0, chromosome_size)

        newborn = []
        for i in range(chromosome_size):
            if i >= cross_over_point:
                newborn.append(parent2[i])
            else:
                newborn.append(parent1[i])

        new_population[ind1] = newborn
        parents.append(ind2)

    if len(parents) == 1:
        parent1 = temp[parents.pop()]
        parent2 = temp[first]
        cross_over_point = np.random.randint(0, chromosome_size)
        newborn = []
        for i in range(chromosome_size):
            if i >= cross_over_point:
                newborn.append(parent2[i])
            else:
                newborn.append(parent1[i])

        new_population[first] = newborn

    # Mutation

    num_genes = chromosome_size * population_size
    num_mutations = np.floor(num_genes * mutation_rate).astype("int")

    for _ in range(num_mutations):
        r = np.random.randint(0, num_genes)
        new_population[r // chromosome_size][r % chromosome_size] = np.random.randint(
            0, 31
        )

    return new_population


# Testing

test_population = np.array(
    [
        [28, 24, 23, 18],
        [8, 22, 14, 25],
        [17, 3, 21, 27],
        [7, 30, 22, 26],
        [8, 8, 28, 23],
        [3, 29, 25, 12],
        [27, 28, 12, 18],
        [5, 6, 10, 14],
        [26, 22, 25, 13],
        [19, 10, 6, 10],
    ]
)

for i in range(num_generations):
    test_population = successor(test_population)

print(test_population)

import numpy as np


class NSGA:

    def __init__(self, functions, population_size=10, num_objectives=2, chromosome_size=2, cross_over_rate=0.6, mutation_rate=0.6, eta=8) -> None:
        self.functions = functions
        self.population_size = population_size
        self.chromosome_size = chromosome_size
        self.cross_over_rate = cross_over_rate
        self.num_objectives = num_objectives
        self.mutation_rate = mutation_rate
        self.eta = eta

    """
    p dominates q iff at least one objective in p is strictly less than
    the corresponding one in q, and all other objectives are at most that of q
    """

    def generate_population(self):
        return np.array(
            [np.random.rand(self.population_size)
             for _ in range(self.chromosome_size)]
        ).T

    def dominates(self, a, b):
        p = np.array([f(a) for f in self.functions])
        q = np.array([f(b) for f in self.functions])
        return (p <= q).all() and (p < q).sum() > 0
        # checks all objectives are <= and at least one is <

    def crowding_distance(self, population, fronts):
        # Based on width of a hyperrectangle
        # tells you how crowded each point of the front is in the solution space
        # Higher value means less crowded
        cd = {i: 0 for i in range(len(population))}
        for front in fronts:
            for f in self.functions:
                front.sort(key=lambda i: f(population[i]))
                cd[front[0]] = cd[front[-1]] = float("inf")
                for i in range(1, len(front) - 1):
                    spread = f(population[front[-1]]) - f(population[front[0]])
                    if spread != 0:
                        cd[front[i]] += (
                            f(population[front[i + 1]]) -
                            f(population[front[i - 1]])
                        ) / (f(population[front[-1]]) - f(population[front[0]]))
                    else:
                        cd[front[i]] = 0

        return cd

    def better(self, i, j, ranks, cd):
        if ranks[i] > ranks[j]:
            return True
        elif ranks[i] < ranks[j]:
            return False
        elif cd[i] > cd[j]:
            return True
        elif cd[j] > cd[i]:
            return False
        else:
            return np.random.rand() > 0.5

    def tournament_selection(self, population, ranks, dist):
        # Binary tournament selection
        layout = np.array([i for i in range(len(population))])
        new_population = []
        for _ in range(2):  # do it twice to keep same population size
            np.random.shuffle(layout)
            for i in range(0, len(population), 2):
                if self.better(layout[i], layout[i + 1], ranks, dist):
                    new_population.append(population[i])
                else:
                    new_population.append(population[i + 1])

        return np.array(new_population)

    def nd_sort(self, population):
        # fast non dominated sorting algorithm
        dominated_by = [set() for _ in range(len(population))]
        ranks = {i: -1 for i in range(len(population))}
        fronts = [[]]
        num_dominating = [0 for _ in range(len(population))]
        for i in range(len(population)):
            p = population[i]
            for j in range(len(population)):
                q = population[j]
                if self.dominates(p, q):
                    dominated_by[i].add(j)
                elif self.dominates(q, p):
                    num_dominating[i] += 1
            if num_dominating[i] == 0:  # non dominant solution found
                ranks[i] = 0
                fronts[0].append(i)

        while fronts[-1]:
            Q = []
            for i in fronts[-1]:
                p = population[i]
                for j in dominated_by[i]:
                    num_dominating[j] -= 1
                    if num_dominating[j] == 0:  # next front element found
                        ranks[j] = len(fronts)
                        Q.append(j)
            fronts.append(Q)
        fronts.pop()

        return ranks, fronts

    def offspring(self, p1, p2):
        c1 = []
        c2 = []
        for i in range(self.chromosome_size):
            # Use inverse cdf of simulated binary crossover operator
            u = np.random.random()
            b = (
                (2 * u) ** (1 / (self.eta + 1))
                if u <= 0.5
                else (1 / (2 * (1 - u))) ** (1 / (self.eta + 1))
            )
            c1.append(0.5 * ((p1[i] + p2[i]) - b * abs(p2[i] - p1[i])))
            c2.append(0.5 * ((p1[i] + p2[i]) + b * abs(p2[i] - p1[i])))
        return np.array(c1), np.array(c2)

    def cross_over(self, population):
        new_population = []
        np.random.shuffle(population)

        for i in range(0, len(population), 2):
            cross = np.random.random() < self.cross_over_rate
            p1, p2 = population[i], population[i + 1]
            if cross:
                c1, c2 = self.offspring(p1, p2)
                new_population.append(c1)
                new_population.append(c2)
            else:
                new_population.append(p1)
                new_population.append(p2)

        return np.array(new_population)

    def mutate(self, population):
        U = np.array([max(decision) for decision in population.T])
        L = np.array([min(decision) for decision in population.T])
        ranges = U - L
        for i in range(len(population)):
            if np.random.random() < self.mutation_rate:
                sol = population[i]
                for j in range(self.chromosome_size):
                    # Polynomial mutation
                    r = np.random.random()
                    delta = (
                        (2 * r) ** (1 / (self.eta + 1)) - 1
                        if r < 0.5
                        else 1 - 2 * (1 - r) ** (1 / (self.eta + 1))
                    )
                    sol[j] += ranges[j] * delta

    def survive(self, population):
        _, fronts = self.nd_sort(population)
        cd = self.crowding_distance(population, fronts)
        i = 0
        size = 0
        new_population = []
        while len(fronts[i]) + size <= self.population_size:
            front = fronts[i]
            for j in range(len(front)):
                new_population.append(population[front[j]])
            size += len(fronts[i])
            i += 1
        if size < self.population_size:
            fronts[i].sort(key=lambda x: cd[x], reverse=True)
            j = 0
            while size + j < self.population_size:
                new_population.append(population[fronts[i][j]])
                j += 1
        return np.array(new_population)

    def evolve(self, population):
        ranks, fronts = self.nd_sort(population)
        cd = self.crowding_distance(population, fronts)
        children = self.tournament_selection(population, ranks, cd)
        children = self.cross_over(population)
        merged = np.concatenate([population, children])
        return self.survive(merged)

    def avg_objective(self, population, i):
        f = self.functions[i]
        return np.mean(np.array([f(c) for c in population]))

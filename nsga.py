import numpy as np
import matplotlib.pyplot as plt
import random
from data import util
import sqlite3

population_size = 10 # ideally even size to allow for simple tournament selection
chromosome_size = 2
cross_over_rate = 0.8
num_objectives = 2
mutation_rate = 0.5
eta = 20

def generate_population(): 
    return np.array([np.random.rand(population_size),np.random.rand(population_size)]).T

# Objective functions
functions = [lambda x : x[0], lambda x : 1 + x[1] - x[0]*x[0]]
# Our goal is to minimise these

"""
p dominates q iff at least one objective in p is strictly less than
the corresponding one in q, and all other objectives are at most that of q
"""

def dominates(a, b): 
    p = np.array([f(a) for f in functions])
    q = np.array([f(b) for f in functions])
    return (p <= q).all() and (p < q).sum() > 0
    # checks all objectives are <= and at least one is <

def crowding_distance(population, fronts):
    # tells you how crowded each point of the front is in the solution space
    cd = {i:0 for i in range(len(population))}
    for front in fronts:
        for f in functions:
            front.sort(key = lambda i : f(population[i]))
            cd[front[0]] = cd[front[-1]] = float('inf')
            for i in range(1, len(front)-1):
                cd[front[i]] += (f(population[front[i+1]]) - f(population[front[i-1]]))/(f(population[front[-1]]) - f(population[front[0]]))
        
    return cd

def better(i, j, ranks, cd):
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

def tournament_selection(population, ranks, dist):
    # Binary tournament selection
    layout = np.array([i for i in range(len(population))])
    new_population = []
    for _ in range(2): # do it twice to keep same population twice
        np.random.shuffle(layout)
        for i in range(0, len(population), 2):
            if better(layout[i], layout[i+1], ranks, dist):
                new_population.append(population[i])
            else:
                new_population.append(population[i+1])

    return np.array(new_population)

def nd_sort(population):
    # fast non dominated sorting algorithm
    dominated_by = [set() for _ in range(len(population))]
    ranks = {i:-1 for i in range(len(population))}
    fronts = [[]]
    num_dominating = [0 for _ in range(len(population))]
    for i in range(len(population)):
        p = population[i]
        for j in range(len(population)):
            q = population[j]
            if dominates(p, q):
                dominated_by[i].add(j)
            elif dominates(q, p):
                num_dominating[i] += 1
        if num_dominating[i] == 0: # non dominant solution found
            ranks[i] = 0
            fronts[0].append(i)
    
    while fronts[-1]:
        Q = []
        for i in fronts[-1]:
            p = population[i]
            for j in dominated_by[i]:
                num_dominating[j] -= 1
                if num_dominating[j] == 0: # next front element found
                    ranks[j] = len(fronts)
                    Q.append(j)
        fronts.append(Q)   
    fronts.pop()

    return ranks, fronts

def offspring(p1, p2):
    c1 = []
    c2 = []
    for i in range(num_objectives):
        # Use inverse cdf of simulated binary crossover operator
        u = np.random.random()
        b = (2*u)**(1/(eta+1)) if u <= 0.5 else (1/(2*(1-u)))**(1/(eta+1))
        c1.append(0.5*((p1[i] + p2[i]) - b*abs(p2[i] - p1[i])))
        c2.append(0.5*((p1[i] + p2[i]) + b*abs(p2[i] - p1[i])))
    return np.array(c1), np.array(c2)

def cross_over(population):
    new_population = []
    np.random.shuffle(population)

    for i in range(0, len(population), 2):
        cross = np.random.random() < cross_over_rate
        p1, p2 = population[i], population[i+1]
        if cross:
            c1, c2 = offspring(p1, p2)
            new_population.append(c1)
            new_population.append(c2)
        else:
            new_population.append(p1)
            new_population.append(p2)

    return np.array(new_population)

def mutate(population):
    U = np.array([max(decision) for decision in population.T])
    L = np.array([min(decision) for decision in population.T])
    ranges = U-L
    for i in range(len(population)):
        if np.random.random() < mutation_rate:
            sol = population[i]
            for j in range(chromosome_size):
                # Polynomial mutation 
                r = np.random.random()
                delta = (2*r)**(1/(eta+1)) - 1 if r < 0.5 else 1 - 2*(1-r)**(1/(eta+1))
                sol[j] += ranges[j] * delta

def survive(population):
    _, fronts = nd_sort(population)
    cd = crowding_distance(population, fronts)
    i = 0
    size = 0
    new_population = []
    while len(fronts[i]) + size <= population_size:
        front = fronts[i]
        for j in range(len(front)):
            new_population.append(population[front[j]]) 
        size += len(fronts[i])
        i += 1
    if size < population_size:
        random.shuffle(fronts[i])
        fronts[i].sort(key=lambda x : cd[x], reverse=True)
        j = 0
        while size+j < population_size:
            new_population.append(population[fronts[i][j]])
            j += 1
    return np.array(new_population)

def evolve(population):


    ranks, fronts = nd_sort(population)
    cd = crowding_distance(population, fronts)
    children = tournament_selection(population, ranks, cd)
    children = cross_over(population)
    merged = np.concatenate([population, children])
    return survive(merged)


def plot():
    population = generate_population()
    for _ in range(20):
        population = evolve(population)
    print(population)
    # obj1 = np.array([functions[0](chromosome) for chromosome in population])
    # obj2 = np.array([functions[1](chromosome) for chromosome in population])
    # plt.scatter(obj1, obj2)
    # plt.show()

# Step 1: Create a SQLite database connection
db_conn = sqlite3.connect('PairsTradingData.db')

# Step 2: Instantiate the OHLCV class with the database connection
ohlc_reader = util.OHLCV(db_conn)

# Step 3: Call the load method with the required parameters
pairs = ['ETHUSDT']
exchanges = ['binance']
since_date = '2024-01-01 00:02:00'

data_frames = ohlc_reader.load(pairs, exchanges, since_date)
df = data_frames[0]

print(df['close'])

plot()
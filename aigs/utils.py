import numpy as np
from tqdm import tqdm


# Problem specific functions:
def sample_random():
    return np.random.uniform(0, 1, (2,))


def evaluate(x):
    fitness = np.exp(-np.std(x))
    behavior = [np.mean(x)]
    return fitness, behavior


# MAP-Elites auxiliary functions:
def get_key(b, resolution):
    # suppose that b is in [0, 1]*
    return tuple(
        [int(x * resolution) if x < 1 else (resolution - 1) for x in b]
    )  # edge case when the behavior is exactly the bound you put it with the previous cell


def iso_line_dd(p1, p2, iso_sigma=0.01, line_sigma=0.2):
    # suppose that the search space is in [0, 1]*
    candidate = p1 + np.random.normal(0, iso_sigma) + np.random.normal(0, line_sigma) * (p2 - p1)
    return np.clip(candidate, np.zeros(p1.shape), np.ones(p1.shape))


def variation_operator(Archive):
    keys = list(Archive.keys())
    key1 = keys[np.random.randint(0, len(keys))]
    key2 = keys[np.random.randint(0, len(keys))]
    return iso_line_dd(Archive[key1]["solution"], Archive[key2]["solution"])


# MAP-Elites hyperparameters
n_budget = 100  # total number of evaluations
n_init = int(0.1 * n_budget)  # number of random solutions to start filling the archive
resolution = 10  # number of cells per dimension

# MAP-Elites:
Archive = {}  # empty archive
for i in tqdm(range(n_budget)):
    if i < n_init:  # initialize with random solutions
        candidate = sample_random()
    else:  # mutation and/or crossover
        candidate = variation_operator(Archive)
    f, b = evaluate(candidate)
    key = get_key(b, resolution)  # get the index of the niche/cell
    if key not in Archive or Archive[key]["fitness"] < f:  # add if new behavior or better fitness
        Archive[key] = {"fitness": f, "behavior": b, "solution": candidate}

print(Archive)

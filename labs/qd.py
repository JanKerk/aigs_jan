# %% qd.py
#   quality diversity exercises
# by: Noah Syrkis

# Imports
import numpy as np
from functools import partial
import matplotlib.pyplot as plt


# %% n-dimensional function with a strange topology
@partial(np.vectorize, signature="(d)->()")
def griewank_function(x):  # this is kind of our fitness function (except we a minimizing)
    return 1 + np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, x.size + 1))))

@partial(np.vectorize, signature="(d)->()")
def sphere_function(x):
    return np.sum(x**2)


@partial(np.vectorize, signature="(d)->(d)", excluded=[0]) # sigma is not getting vectorized
def mutate(sigma, x):  # Adding random noise drawn from a normal distribution --> random variation
    return x + np.random.normal(0, sigma, x.shape)


@partial(np.vectorize, signature="(d),(d)->(d)")
def crossover(x1, x2):  # New vector (child) lies somewhere between x1 and x2 (parents)
    return x1 * np.random.rand() + x2 * (1 - np.random.rand()) # parents get combined as a weighted average

"""
def step(x, cfg):
    fitness = griewank_function(x)
    idxs = np.argsort(fitness)[: int(cfg.population * cfg.proportion)]  # select best
    seed = np.tile(x[idxs], (int(cfg.population * cfg.proportion), 1))  # cross over
    x = crossover(seed, seed[np.random.permutation(seed.shape[0])])  # mutate
    return mutate(cfg.sigma, x), fitness  # return new generation and fitness
"""

def step(x, cfg):
    loss = sphere_function(x)
    idxs = np.argsort(loss)[: int(cfg.population * cfg.proportion)] # select best
    best = np.tile(x[idxs], (int(cfg.population * cfg.proportion), 1)) # copy best to get 100% population with best again
    x = crossover(best, best[np.random.permutation(best.shape[0])])  # crossover: every entry in best with another entry from best, but second input is shuffled before
    return mutate(cfg.sigma, x), loss  # return new generation and loss

# %% Setup

def main(cfg):
    x = np.random.rand(cfg.population, cfg.dimensions)
    fitnesses = []
    for gen in range(cfg.generation):
        x, fitness = step(x, cfg)
        fitnesses.append(fitness.min())
        print(f"Generation {gen}: Best fitness = {fitness.min()}")

    plt.plot(fitnesses)
    plt.yscale("log")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.show()
    
    # Boxplot to see fitness distribution
    plt.boxplot(fitnesses, showfliers=False)
    plt.yscale("log")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Distribution")
    plt.show()
    
    # Plot function
    plot(sphere_function)


# %% Plotting function that I think we should put in utils.py
def plot(fn):
    x1 = np.linspace(-10, 10, 100)
    x2 = np.linspace(-10, 10, 100)
    xs = np.stack(np.meshgrid(x1, x2), axis=-1)
    ys = fn(xs)
    plt.imshow(ys, cmap="viridis")
    plt.colorbar()
    plt.show()


env, pop = init(ctx.config)
from pcgym.envs import PcgrlEnv
from typing import Tuple
from pcgym.envs.helper import get_string_map

# import qdax
# from qdax.core.map_elites import MAPElites
# from PIL import Image
# from qdax.core.emitters.mutation_operators import polynomial_mutation
# env._rep._mep = pop[0]
# Image.fromarray(env.render()).save("map.png")
# fitness, behavior = map(np.array, zip(*list(map(lambda p: eval(env, p), pop))))
# print(behavior.shape, fitness.shape)
# # print(fitness)
# # print(behavior)
# # print(list(map(lambda p: mutate(ctx.config, env, p), pop)))


# # %% Init population (maps)
# def init(cfg) -> Tuple[PcgrlEnv, np.ndarray]:
#     env = PcgrlEnv(prob=cfg.qd.game, rep=cfg.qd.rep, render_mode="rgb_array")
#     env.reset()
#     pop = np.random.randint(0, env.get_num_tiles(), (cfg.qd.n, *env._rep._map.shape))  # type: ignore
#     return env, pop


# # %% using gym-pcg to evaluate map quality
# def eval(env, p) -> Tuple[int, np.ndarray]:
#     env._rep._map = p
#     string_map = get_string_map(env._rep._map, env._prob.get_tile_types())
#     stats = env._prob.get_stats(string_map)
#     behavior = np.array([stats["disjoint-tubes"], stats["empty"]])
#     return env._prob.get_reward(stats, {k: 0 for k in stats.keys()}), behavior


# def mutate(cfg, env, p) -> np.ndarray:
#     mask = np.random.random(p.shape) < cfg.qd.p
#     p[mask] = np.random.randint(0, env.get_num_tiles(), p[mask].shape)
#     return p


# def archive(state, p):
#     raise NotImplementedError

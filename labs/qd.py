# %% qd.py
#   quality diversity exercises
# by: Noah Syrkis

# Imports
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from typing import Tuple
import pcgym
from pcgym import PcgrlEnv
from PIL import Image
from tqdm import tqdm


# %% n-dimensional function with a strange topology
@partial(np.vectorize, signature="(d)->()")
def griewank_function(x):  # this is kind of our fitness function (except we are minimizing)
    return 1 + np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, x.size + 1))))

# Sphere function
@partial(np.vectorize, signature="(d)->()")
def sphere_function(x):
    return np.sum(x**2)


@partial(np.vectorize, signature="(d)->(d)", excluded=[0]) # sigma is not getting vectorized
def mutate(sigma, x):  # Adding random noise drawn from a normal distribution --> random variation
    return x + np.random.normal(0, sigma, x.shape)


@partial(np.vectorize, signature="(d),(d)->(d)")
def crossover(x1, x2):  # New vector (child) lies somewhere between x1 and x2 (parents)
    return x1 * np.random.rand() + x2 * (1 - np.random.rand()) # parents get combined as a weighted average

""""
def step(x, cfg):
    fitness = griewank_function(x)
    idxs = np.argsort(fitness)[: int(cfg.population * cfg.proportion)]  # select best
    seed = np.tile(x[idxs], (int(cfg.population * cfg.proportion), 1))  # cross over
    x = crossover(seed, seed[np.random.permutation(seed.shape[0])])  # mutate
    return mutate(cfg.sigma, x), fitness  # return new generation and fitness


def step(x, cfg):
    loss = sphere_function(x)
    idxs = np.argsort(loss)[: int(cfg.population * cfg.proportion)] # select best
    best = np.tile(x[idxs], (int(cfg.population * cfg.proportion), 1)) # copy best to get 100% population with best again
    x = crossover(best, best[np.random.permutation(best.shape[0])])  # crossover: every entry in best with another entry from best, but second input is shuffled before
    return mutate(cfg.sigma, x), loss  # return new generation and loss """

# %% Setup

def main(cfg):
    # Code for EA
    """ x = np.random.rand(cfg.population, cfg.dimensions)
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
    plt.xlabel("Generation")
    plt.yscale("log")
    plt.ylabel("Fitness Distribution")
    plt.show()
    
    # Plot function
    plot(sphere_function) """
    
    """
    # Just random map
    env, pop = init_pcgym(cfg=cfg)
    env._rep._map = pop[0]
    Image.fromarray(obj=env.render()).save(fp="map.png")
    map = get_string_map(map=env._rep._map, tiles = env._prob.get_tile_types())
    stats = env._prob.get_stats(map)
    print(stats)
    """
    # Generate random map
    env, pop = init_pcgym(cfg=cfg)
    
    # MAP-Elites hyperparameters
    n_budget = 1000  # total number of evaluations
    n_init = min(int(0.1 * n_budget), len(pop))  # number of random solutions to start filling the archive
    resolution = 10  # number of cells per dimension

    # MAP-Elites:
    Archive = {}  # empty archive
    for i in tqdm(range(n_budget)):
        if i < n_init:  # initialize with random solutions (and normalize)
            candidate = pop[i]/ (env.get_num_tiles() - 1)
        else:  # mutation and/or crossover
            candidate = variation_operator(Archive)
            
        # Convert to integer tiles and then to string format for evaluation
        level_int = np.round(candidate * (env.get_num_tiles() - 1)).astype(int)
        
        # Convert to string format that get_stats expects
        tile_types = env._prob.get_tile_types()
        level_str = [[tile_types[tile_int] for tile_int in row] for row in level_int]
        
        f, b = evaluate(level_str, env)
        key = get_key(b, resolution)  # get the index of the niche/cell
        if key not in Archive or Archive[key]["fitness"] < f:  # add if new behavior or better fitness
            Archive[key] = {"fitness": f, "behavior": b, "solution": level_int.copy()}

    print(Archive)
    
    # Print Image of best fitness level
    best_entry = max(Archive.values(), key=lambda x: x["fitness"])
    best_level = best_entry["solution"]
    env._rep._map = best_level
    env.reset()
    Image.fromarray(obj=env.render()).save(fp="best_map.png")
    exit()
    
    # Let a random agent play
    # score = random_agent(env=env)
    # print(f"Random agent total reward: {score: .3f}")
    # exit()


# %% Plotting function that I think we should put in utils.py
def plot(fn):
    x1 = np.linspace(-10, 10, 100)
    x2 = np.linspace(-10, 10, 100)
    xs = np.stack(np.meshgrid(x1, x2), axis=-1)
    ys = fn(xs)
    plt.imshow(ys, cmap="viridis")
    plt.colorbar()
    plt.show()


# %% Init population (maps)
def init_pcgym(cfg) -> Tuple[PcgrlEnv, np.ndarray]:
    env = PcgrlEnv(prob=cfg.game, rep=cfg.rep, render_mode="rgb_array")
    env.reset()
    pop = np.random.randint(0, env.get_num_tiles(), (cfg.n, *env._rep._map.shape))  # type: ignore
    return env, pop

# Random agent in pcgym
def random_agent(env, steps=1000, render=True):
    frames = []
    env.reset()
    total_reward = 0.0
    for i in range(steps):
        frames.append(Image.fromarray(obj=env.render())) # Collecting frames
        
        a = env.action_space.sample() # Choosing a random action
        obs, reward, terminated, truncated, info = env.step(a) # step function
        total_reward+= reward
        if terminated or truncated:
            print(f"Episode ended after {i+1} timesteps.")
            break
    
    # Seeing agent in action
    frames[0].save(
        "random_agent.gif",
        save_all = True,
        append_images = frames[1:],
        duration = 100,
        loop = 0
    )
    
    return total_reward
        
def fitness(stats):
    # Weights
    playability = 0.6
    challenge = 0.4
    
    # Playability score
    playability_score = 0.0
    
    # Level must be completable
    if stats['dist-win'] == 0:  # Mario reaches the end
        playability_score += 0.8
    else:
        # Penalty based on how far from completion
        penalty = min(stats['dist-win'] / 100.0, 0.8)
        playability_score += 0.8 - penalty
    
    # No disjoint tubes
    if stats['disjoint-tubes'] == 0:
        playability_score += 0.2
    
    # Challenge Score  
    challenge_score = 0.0
    
    # Enemy count
    enemies = stats['enemies']
    if 10 <= enemies <= 30:
        challenge_score += 0.5
    elif enemies < 10:
        challenge_score += 0.5 * (enemies / 10.0)
    else:
        challenge_score += 0.5 * max(0, 1.0 - (enemies - 30) / 30.0)
    
    # Jump count
    jumps = stats['jumps']
    if jumps >= 20:
        challenge_score += 0.3
    else:
        challenge_score += 0.3 * (jumps / 20.0)
    
    # Jump distance
    if stats['jumps-dist'] <= 6:
        challenge_score += 0.2
    else:
        challenge_score += 0.2 * max(0, 1.0 - (stats['jumps-dist'] - 6) / 6.0)
    
    # Final fitness
    fitness = playability * playability_score + challenge * challenge_score
    
    return max(0.0, min(1.0, fitness))

def get_behaviour(stats):
    return (stats['noise'], stats['empty'])
        
########## MAP-Elites ##########
# Problem specific functions:
def sample_random():
    return np.random.uniform(0, 1, (2,))


def evaluate(x, env):
    stats = env._prob.get_stats(x)
    
    # Inverse distance from goal
    if stats['dist-win'] == 0:
        fitness = 1.0  # Perfect completion
    else:
        # Convert distance to fitness
        fitness = max(0.0, 1.0 - stats['dist-win'] / 114.0)
    
    # Behavior: number of jumps and jump distance (normalize)
    behavior = [min(stats['jumps'] / 30.0, 1.0), min(stats['jumps-dist'] / 4.0, 1.0)]
    
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

#########################################################################
##  BELOW HERE THERE BE DRAGONS (left over stuff i played around with) ##
#########################################################################

# %% Plotting function that I think we should put in utils.py
# def our_plot_function(fn):
#     x1 = np.linspace(-10, 10, 100)
#     x2 = np.linspace(-10, 10, 100)
#     xs = np.stack(np.meshgrid(x1, x2), axis=-1)
#     ys = fn(xs)
#     plt.imshow(ys, cmap="viridis")
#     plt.colorbar()
#     plt.show()


# env, pop = init(ctx.config)
# from pcgym.envs import PcgrlEnv
# from typing import Tuple
# from pcgym.envs.helper import get_string_map

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
# imports
from __future__ import annotations
import numpy as np
import aigs
from aigs import State, Env
from dataclasses import dataclass, field
import random
import math


# %% Setup
env: Env


# %%

def minimax(state: State, maxim: bool) -> int:
    if state.ended:
        return state.point

    temp = -10 if maxim else 10
    for action in np.where(state.legal)[0]:
        value = minimax(env.step(state, action), not maxim)
        temp = max(temp, value) if maxim else min(temp, value)
    return temp


def alpha_beta(state: State, maxim: bool, alpha: int = -100000000000, beta: int = 100000000000) -> int:
    if state.ended:
        return state.point
    
    # Maximise if it is current player
    if maxim:
        for action in np.where(state.legal)[0]:
            value = alpha_beta(env.step(state, action), not maxim, alpha, beta)
            # Check if value is better than alpha
            if value > alpha:
                alpha = value
            # Prune if true --> opponent will avoid this branch
            if alpha >= beta:
                break
        return alpha
    
    # Minimise if it is the opponent of current player
    else:
        for action in np.where(state.legal)[0]:
            value = alpha_beta(env.step(state, action), maxim, alpha, beta)
            # Check if value is better than beta
            if value < beta:
                beta = value
            # Prune if alpha is already better
            if alpha >= beta:
                break
        return beta

# Heuristic weights
win_val = 10000       # terminal wins/losses always dominates
open3 = 100           # 3-in-a-row with 1 empty
open2 = 10            # 2-in-a-row with 2 empties
open1 = 1             # 1-in-a-row with 3 empties
block3 = 120          # opponent 3+empty -> value blocking
block2 = 8            # opponent 2+two empties --> light penalty
center_val = 3        # prefer center control a bit

def windows(board):
    windows = []
    Rows, Columns = board.shape
    # horizontal
    for r in range(Rows):
        for c in range(Columns - 3):
            windows.append(board[r, c:c+4])
    # vertical
    for r in range(Rows - 3):
        for c in range(Columns):
            windows.append(board[r:r+4, c])
    # diagonal \
    for r in range(Rows - 3):
        for c in range(Columns - 3):
            windows.append(np.array([board[r+i, c+i] for i in range(4)]))
    # diagonal /
    for r in range(Rows - 3):
        for c in range(Columns - 3):
            windows.append(np.array([board[r+3-i, c+i] for i in range(4)]))
    return windows

def heuristic_minimax(state: State, maxim: bool, depth: int = 5) -> int:
    if state.ended:
        # Real wins/losses trump any heuristic values
        return win_val * state.point  # point in {+1, 0, -1}
    
    if depth == 0:
        board = state.board
        Rows, Columns = board.shape
        player  = 1 if maxim else -1
        opp = -player
        score = 0
        

        # Small center preference
        center_col = board[:, Columns // 2]
        score += center_val * np.count_nonzero(center_col == player)
        score -= center_val * np.count_nonzero(center_col == opp)

        # Scan all potential 4-in-a-row windows
        for w in windows(board):
            # Count how many pieces who has within this window
            player_count  = np.count_nonzero(w == player)
            opp_count = np.count_nonzero(w == opp)
            empty_count = np.count_nonzero(w == 0)

            # Only windows which can become a win (so no windows with pieces of both player and opponent)
            if opp_count == 0:
                # player windows
                if player_count == 3 and empty_count == 1:
                    score += open3
                elif player_count == 2 and empty_count == 2:
                    score += open2
                elif player_count == 1 and empty_count == 3:
                    score += open1
            elif player_count == 0:
                # opponent windows
                if opp_count == 3 and empty_count == 1:
                    score -= block3
                elif opp_count == 2 and empty_count == 2:
                    score -= block2
        return score
    
    temp = -10 if maxim else 10
    for action in np.where(state.legal)[0]:
        value = heuristic_minimax(env.step(state, action), not maxim, depth-1)
        temp = max(temp, value) if maxim else min(temp, value)
    return temp

@dataclass
class Node:
    state: State
    parent: Node = None        # Parent node
    children: list = None      # Children nodes
    visits: int = 0            # Number of times the node has been visited
    value: int = 0             # Accumulated value from simulations
    tried_actions: list = None # Tried actions for this node
    action: int = 0            # Store the action that led to this node

    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.tried_actions is None:
            self.tried_actions = []

# Intuitive but difficult in terms of code
def monte_carlo(state: State, cfg) -> int:
    root_node = Node(state=state)
    node = root_node
    
    for n in range(cfg.max_iterations):
        
        # 1. Selection
        node = tree_policy(node, cfg)
        
        # 3. Simulation
        result = default_policy(node.state)
        
        # 4. Backpropagation
        backup(node, result)
    
    # Look for the child with the best value and return the action to get to this state
    best_node = best_child(root_node, cfg.c)
    best_action = best_node.action
    
    return best_action

def tree_policy(node: Node, cfg) -> Node:
    while not node.state.ended:
        
        # 2. Expansion
        expanded_node = expand(node)
        # Check if all actions were already tried for this node
        if expanded_node is not node:
            node = expanded_node
        else:
            # Choose child with maximum UCT
            node = best_child(node, cfg.c)
    return node


def expand(v: Node) -> Node:
    # Collect all possible actions for this node
    actions = np.where(v.state.legal)[0]
    
    # If there are no actions
    if not actions.any():
        return v
    
    # Choose a random action
    action = random.choice(actions)
    
    # Check if the action was already tried
    if action in v.tried_actions:
        new_action = random.choice(actions)
        while new_action == action:
            new_action = random.choice(actions)
        action = new_action
        v.tried_actions.append(action)
    else:
        v.tried_actions.append(action)
    
    # Add new child node/Expand tree
    new_state = env.step(v.state, action)
    child = Node(state=new_state, parent=v)
    child.action = action
    v.children.append(child)
    return child


def best_child(root: Node, c) -> Node:
    best = None
    best_value = float('-inf')
    for child in root.children:
        # UCT formula
        value = child.value / (child.visits+1) + 2 * c * math.sqrt(math.log(root.visits+1) / (child.visits+1))
        if value > best_value:
            best_value = value
            best = child
    return best


def default_policy(state: State) -> int:
    # Run random simulation
    current_state = state
    while not current_state.ended:
        # Collect all possible actions for this node
        actions = np.where(current_state.legal)[0]
        # Choose a random action
        action = random.choice(actions)
        current_state = env.step(current_state, action)
    return current_state.point


def backup(node, delta) -> None:
    while node is not None:
        node.visits+=1
        if node.state.maxim:
            node.value+=delta
        else:
            node.value-=delta
        node = node.parent


# Main function
def main(cfg) -> None:
    global env
    env = aigs.make(cfg.game)
    state = env.init()

    while not state.ended:
        actions = np.where(state.legal)[0]  # the actions to choose from

        match getattr(cfg, state.player):
            case "random":
                a = np.random.choice(actions).item()

            case "human":
                print(state, end="\n\n")
                a = int(input(f"Place your piece ({'x' if state.minim else 'o'}): "))

            case "minimax":
                values = [minimax(env.step(state, a), not state.maxim) for a in actions]
                a = actions[np.argmax(values) if state.maxim else np.argmin(values)]
                
            case "heuristic":
                values = [heuristic_minimax(env.step(state, a), not state.maxim) for a in actions]
                a = actions[np.argmax(values) if state.maxim else np.argmin(values)]
                
            case "alpha_beta":
                values = [alpha_beta(env.step(state, a), not state.maxim, -1, 1) for a in actions]
                a = actions[np.argmax(values) if state.maxim else np.argmin(values)]

            case "monte_carlo":
                a = monte_carlo(state, cfg)

            case _:
                raise ValueError(f"Unknown player {state.player}")

        state = env.step(state, a)

    print(f"{['nobody', 'o', 'x'][state.point]} won", state, sep="\n")

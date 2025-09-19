# games.py
#   aigs games
# by: Noah Syrkis

# imports
from aigs.types import State, Env
import numpy as np


# connect four
class ConnectFour(Env):
    def init(self) -> State:
        board = np.zeros((6, 7), dtype=int)
        legal = board[0] == 0
        state = State(board=board, legal=legal)
        return state

    def step(self, state, action) -> State:
        # hint: use x.diagonal(i)
        
        # make your move
        board = state.board.copy()

        for i, val in enumerate(board[::-1, action]):
            if val == 0:
                row_index = 6 - 1 - i
                board[row_index, action] = 1 if state.maxim else -1
                break
            elif i == 5: # if the column is already full
                raise AssertionError(f"Invalid move: {action}")
        
        # function for checking vector
        def check_vector(v):
            if len(v) < 4:
                return False
            for i in range(len(v)-3):
                if v[i] and v[i+1] and v[i+2] and v[i+3]:
                    return True
            return False
        
        # was it a winning move?
        mask = board == (1 if state.maxim else -1)
        
        rows = [rows for rows in mask]
        cols = [cols for cols in mask.T]
        diagonals = [mask.diagonal(i) for i in range(-5, 7)]
        flipped_diag = [np.fliplr(mask).diagonal(i) for i in range(-5, 7)]
        
        checks = [check_vector(v) for v in rows + cols + diagonals + flipped_diag]
        
        winner = True in checks
        legal = board[0] == 0
        point = (1 if state.maxim else -1) if winner else 0  
        
        # return the next state
        return State(
            board=board,
            legal = legal,  # empty board positions
            ended= (not legal.any()) or winner,
            point=point,
            maxim=not state.maxim,
        )


# tic tac toe
class TicTacToe(Env):
    def init(self) -> State:
        board = np.zeros((3, 3), dtype=int)
        legal = board.flatten() == 0
        state = State(board=board, legal=legal)
        return state

    def step(self, state, action) -> State:
        # make your move
        board = state.board.copy()
        assert board[action // 3, action % 3] == 0, f"Invalid move: {action}"
        board[action // 3, action % 3] = 1 if state.maxim else -1

        # was it a winning move?
        mask = board == (1 if state.maxim else -1)
        winner: bool = (
            mask.all(axis=1).any()  # |
            or mask.all(axis=0).any()  # —
            or mask.trace() == 3  # \
            or mask.T.trace() == 3  # /
        )

        # return the next state
        return State(
            board=board,
            legal=board.flatten() == 0,  # empty board positions
            ended=(board != 0).all() | winner,
            point=(1 if state.maxim else -1) if winner else 0,
            maxim=not state.maxim,
        )

"""Finite RiverSwim MDP used in ECE 414/517 Coding Assignment 1."""

from __future__ import annotations

import numpy as np


LEFT = 0
RIGHT = 1


class RiverSwimMDP:
    """Six-state RiverSwim with configurable current strength.

    Attributes
    ----------
    P : ndarray, shape (num_states, num_actions, num_states)
        Transition probabilities P[s, a, s_next].
    R : ndarray, shape (num_states, num_actions)
        Expected immediate rewards R[s, a].
    """

    _RIGHT_DYNAMICS = {
        "weak": (0.05, 0.60, 0.35),
        "medium": (0.10, 0.65, 0.25),
        "strong": (0.15, 0.75, 0.10),
    }

    def __init__(
        self,
        current: str = "medium",
        num_states: int = 6,
        left_reward: float = 0.005,
        right_reward: float = 1.0,
    ) -> None:
        if current not in self._RIGHT_DYNAMICS:
            choices = ", ".join(self._RIGHT_DYNAMICS)
            raise ValueError(f"current must be one of: {choices}")
        if num_states < 3:
            raise ValueError("num_states must be at least 3")

        self.current = current
        self.num_states = num_states
        self.num_actions = 2
        self.initial_state = 0
        self.P = np.zeros((num_states, self.num_actions, num_states), dtype=float)
        self.R = np.zeros((num_states, self.num_actions), dtype=float)

        # LEFT moves one state downstream; at the boundary it remains in state 0.
        for state in range(num_states):
            self.P[state, LEFT, max(0, state - 1)] = 1.0
        self.R[0, LEFT] = left_reward

        # RIGHT fights the current: move left, stay, or move right.
        p_back, p_stay, p_forward = self._RIGHT_DYNAMICS[current]
        for state in range(num_states):
            left_state = max(0, state - 1)
            right_state = min(num_states - 1, state + 1)
            self.P[state, RIGHT, left_state] += p_back
            self.P[state, RIGHT, state] += p_stay
            self.P[state, RIGHT, right_state] += p_forward
        self.R[num_states - 1, RIGHT] = right_reward

        if not np.allclose(self.P.sum(axis=2), 1.0):
            raise RuntimeError("transition probabilities must sum to one")

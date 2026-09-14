"""A two-state deterministic MDP for debugging dynamic-programming code."""

from __future__ import annotations

import numpy as np


class TinyMDP:
    """State 0 can stay for reward 1 or move to state 1 for reward 0.

    State 1 is absorbing and gives reward 2 under either action. With gamma > 0.5,
    moving from state 0 to state 1 is optimal. Equal-valued actions in state 1
    exercise the required lowest-index tie-breaking rule.
    """

    def __init__(self) -> None:
        self.num_states = 2
        self.num_actions = 2
        self.initial_state = 0
        self.P = np.zeros((2, 2, 2), dtype=float)
        self.R = np.array([[1.0, 0.0], [2.0, 2.0]], dtype=float)
        self.P[0, 0, 0] = 1.0
        self.P[0, 1, 1] = 1.0
        self.P[1, :, 1] = 1.0

"""Starter code for policy iteration and value iteration.

Complete the TODO blocks. Do not change function names, arguments, or return types.
"""

from __future__ import annotations

import numpy as np


def _validate_discount(discount: float) -> None:
    if not 0.0 <= discount < 1.0:
        raise ValueError("discount must satisfy 0 <= discount < 1")


def bellman_backup(mdp, value: np.ndarray, state: int, action: int, discount: float) -> float:
    """Return one state-action Bellman backup Q_V(state, action)."""
    _validate_discount(discount)
    # TODO: combine the immediate reward and discounted expected next-state value.
    raise NotImplementedError


def policy_evaluation(
    mdp,
    policy: np.ndarray,
    discount: float,
    tolerance: float = 1e-8,
    max_iterations: int = 100_000,
) -> np.ndarray:
    """Iteratively evaluate a deterministic policy and return V^policy."""
    _validate_discount(discount)
    policy = np.asarray(policy, dtype=int)
    if policy.shape != (mdp.num_states,):
        raise ValueError("policy must have shape (mdp.num_states,)")
    value = np.zeros(mdp.num_states, dtype=float)

    # TODO: repeatedly apply the policy Bellman operator. Use a copy (synchronous
    # updates), stop when max(abs(new_value - value)) <= tolerance, and raise
    # RuntimeError if max_iterations is reached without convergence.
    raise NotImplementedError


def policy_improvement(mdp, value: np.ndarray, discount: float) -> np.ndarray:
    """Return a deterministic policy greedy with respect to value."""
    _validate_discount(discount)
    value = np.asarray(value, dtype=float)
    if value.shape != (mdp.num_states,):
        raise ValueError("value must have shape (mdp.num_states,)")

    # TODO: compute every action value and use np.argmax for required tie-breaking.
    raise NotImplementedError


def policy_iteration(
    mdp,
    discount: float,
    tolerance: float = 1e-8,
    max_iterations: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (optimal_value, optimal_policy) using policy iteration."""
    _validate_discount(discount)
    policy = np.zeros(mdp.num_states, dtype=int)

    # TODO: alternate policy evaluation and improvement until the policy is stable.
    # Count outer policy-improvement steps against max_iterations.
    raise NotImplementedError


def value_iteration(
    mdp,
    discount: float,
    tolerance: float = 1e-8,
    max_iterations: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (optimal_value, optimal_policy) using value iteration."""
    _validate_discount(discount)
    value = np.zeros(mdp.num_states, dtype=float)

    # TODO: repeatedly apply the optimal Bellman operator with synchronous updates.
    # After convergence, extract a greedy policy with policy_improvement.
    raise NotImplementedError

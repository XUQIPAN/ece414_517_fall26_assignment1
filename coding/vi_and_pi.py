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
    return float(mdp.R[state, action] + discount * np.dot(mdp.P[state, action], value))


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

    states = np.arange(mdp.num_states)
    rewards = mdp.R[states, policy]
    transitions = mdp.P[states, policy]
    for _ in range(max_iterations):
        new_value = rewards + discount * (transitions @ value)
        if np.max(np.abs(new_value - value)) <= tolerance:
            return new_value
        value = new_value
    raise RuntimeError("policy evaluation did not converge within max_iterations")


def policy_improvement(mdp, value: np.ndarray, discount: float) -> np.ndarray:
    """Return a deterministic policy greedy with respect to value."""
    _validate_discount(discount)
    value = np.asarray(value, dtype=float)
    if value.shape != (mdp.num_states,):
        raise ValueError("value must have shape (mdp.num_states,)")

    action_values = mdp.R + discount * (mdp.P @ value)
    return np.argmax(action_values, axis=1)


def policy_iteration(
    mdp,
    discount: float,
    tolerance: float = 1e-8,
    max_iterations: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (optimal_value, optimal_policy) using policy iteration."""
    _validate_discount(discount)
    policy = np.zeros(mdp.num_states, dtype=int)

    for _ in range(max_iterations):
        value = policy_evaluation(mdp, policy, discount, tolerance=tolerance)
        new_policy = policy_improvement(mdp, value, discount)
        if np.array_equal(new_policy, policy):
            return value, new_policy
        policy = new_policy
    raise RuntimeError("policy iteration did not converge within max_iterations")


def value_iteration(
    mdp,
    discount: float,
    tolerance: float = 1e-8,
    max_iterations: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (optimal_value, optimal_policy) using value iteration."""
    _validate_discount(discount)
    value = np.zeros(mdp.num_states, dtype=float)

    for _ in range(max_iterations):
        action_values = mdp.R + discount * (mdp.P @ value)
        new_value = np.max(action_values, axis=1)
        if np.max(np.abs(new_value - value)) <= tolerance:
            return new_value, policy_improvement(mdp, new_value, discount)
        value = new_value
    raise RuntimeError("value iteration did not converge within max_iterations")

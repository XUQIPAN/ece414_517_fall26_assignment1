"""Public tests. Additional valid MDPs and tolerances will be used for grading."""

from __future__ import annotations

import numpy as np

from river_swim import RiverSwimMDP
from tiny_mdp import TinyMDP
from vi_and_pi import (
    bellman_backup,
    policy_evaluation,
    policy_improvement,
    policy_iteration,
    value_iteration,
)


def test_bellman_backup_on_tiny_mdp() -> None:
    mdp = TinyMDP()
    value = np.array([3.0, 4.0])
    assert np.isclose(bellman_backup(mdp, value, 0, 0, 0.9), 3.7)
    assert np.isclose(bellman_backup(mdp, value, 0, 1, 0.9), 3.6)


def test_policy_evaluation_on_tiny_mdp() -> None:
    mdp = TinyMDP()
    policy = np.array([1, 0])
    value = policy_evaluation(mdp, policy, discount=0.9, tolerance=1e-11)
    assert value.shape == (2,)
    assert np.allclose(value, np.array([18.0, 20.0]), atol=1e-7)


def test_policy_improvement_and_tie_breaking() -> None:
    mdp = TinyMDP()
    value = np.array([18.0, 20.0])
    policy = policy_improvement(mdp, value, discount=0.9)
    assert policy.dtype.kind in "iu"
    assert np.array_equal(policy, np.array([1, 0]))


def test_pi_and_vi_agree_on_tiny_mdp() -> None:
    mdp = TinyMDP()
    value_pi, policy_pi = policy_iteration(mdp, discount=0.9, tolerance=1e-11)
    value_vi, policy_vi = value_iteration(mdp, discount=0.9, tolerance=1e-11)
    assert np.allclose(value_pi, value_vi, atol=1e-7)
    assert np.array_equal(policy_pi, policy_vi)
    assert np.array_equal(policy_vi, np.array([1, 0]))


def test_pi_and_vi_agree_on_riverswim() -> None:
    mdp = RiverSwimMDP(current="medium")
    value_pi, policy_pi = policy_iteration(mdp, discount=0.95, tolerance=1e-10)
    value_vi, policy_vi = value_iteration(mdp, discount=0.95, tolerance=1e-10)
    assert value_pi.shape == (mdp.num_states,)
    assert policy_pi.shape == (mdp.num_states,)
    assert np.allclose(value_pi, value_vi, atol=1e-6)
    assert np.array_equal(policy_pi, policy_vi)

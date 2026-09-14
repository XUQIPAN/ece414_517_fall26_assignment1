"""Experiment driver for Coding Assignment 1."""

from __future__ import annotations

import numpy as np

from river_swim import LEFT, RiverSwimMDP
from vi_and_pi import policy_iteration, value_iteration


DISCOUNT_GRID = np.round(np.arange(0.00, 1.00, 0.01), 2)


def largest_left_discount(current: str) -> tuple[float | None, float, float]:
    """Return threshold gamma and PI/VI initial values at gamma=0.99."""
    mdp = RiverSwimMDP(current=current)
    left_discounts: list[float] = []
    for discount in DISCOUNT_GRID:
        value_vi, policy_vi = value_iteration(mdp, float(discount), tolerance=1e-10)
        value_pi, policy_pi = policy_iteration(mdp, float(discount), tolerance=1e-10)
        if not np.allclose(value_vi, value_pi, atol=1e-6):
            raise AssertionError("PI and VI values disagree")
        if not np.array_equal(policy_vi, policy_pi):
            raise AssertionError("PI and VI policies disagree")
        if policy_vi[mdp.initial_state] == LEFT:
            left_discounts.append(float(discount))

    value_vi, _ = value_iteration(mdp, 0.99, tolerance=1e-10)
    value_pi, _ = policy_iteration(mdp, 0.99, tolerance=1e-10)
    threshold = max(left_discounts) if left_discounts else None
    return threshold, float(value_vi[0]), float(value_pi[0])


def main() -> None:
    print("current  largest gamma choosing LEFT   V_VI(s0) at .99   V_PI(s0) at .99")
    for current in ("weak", "medium", "strong"):
        threshold, value_vi, value_pi = largest_left_discount(current)
        threshold_text = "none" if threshold is None else f"{threshold:.2f}"
        print(f"{current:7s}   {threshold_text:>10s}                  {value_vi:10.4f}        {value_pi:10.4f}")


if __name__ == "__main__":
    main()

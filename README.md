# ECE 414/517 - Homework 1 + Coding Assignment 1

Released September 14, 2026. Due Oct 8, 2026 at 11:59 p.m. ET.

Homework 1 and Coding Assignment 1 are separate submissions, each graded out of 100 points.

## Package contents

- `ece414_517_fall26_assignment1.pdf`: compiled student handout.
- `coding/river_swim.py`: RiverSwim MDP implementation (do not modify).
- `coding/tiny_mdp.py`: small deterministic MDP for debugging (do not modify).
- `coding/vi_and_pi.py`: starter file; complete the marked TODO blocks.
- `coding/run_experiments.py`: experiment driver for the required analysis.
- `coding/test_public.py`: public correctness tests.
- `coding/requirements.txt`: minimal Python dependencies.

## Setup and testing

From the package root:

```text
python3 -m venv <environment_name>
source <environment_name>/bin/activate
python -m pip install -r coding/requirements.txt
python -m pytest coding/test_public.py
python coding/run_experiments.py
```

Submit the written PDF and completed `coding/` directory as separate Canvas submissions.
Do not include `.venv`, `__pycache__`, or `.pytest_cache` in the coding ZIP.

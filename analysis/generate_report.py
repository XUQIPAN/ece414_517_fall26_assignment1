"""Reproduce the Coding Assignment 1 analysis and its written PDF.

Run from the repository root with Python 3.10+ and coding/requirements.txt
installed: python analysis/generate_report.py
The final PDF also requires pdflatex; all TeX build files use a temporary folder.
"""

from __future__ import annotations

import atexit
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
_plot_cache = tempfile.TemporaryDirectory(prefix="riverswim_matplotlib_")
atexit.register(_plot_cache.cleanup)
os.environ.setdefault("MPLCONFIGDIR", _plot_cache.name)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "coding"))

from river_swim import LEFT, RIGHT, RiverSwimMDP
from run_experiments import DISCOUNT_GRID
from vi_and_pi import policy_iteration, value_iteration

CURRENTS = ("weak", "medium", "strong")
TOLERANCE = 1e-10
EXTENSION_GRID = np.linspace(0.80, 0.999, 50)


def evaluate_grid(discounts: np.ndarray) -> list[dict]:
    """Check both planners against each other and direct policy evaluation."""
    rows = []
    for current in CURRENTS:
        mdp = RiverSwimMDP(current=current)
        states = np.arange(mdp.num_states)
        for discount in discounts:
            discount = float(discount)
            value_vi, policy_vi = value_iteration(mdp, discount, tolerance=TOLERANCE)
            value_pi, policy_pi = policy_iteration(mdp, discount, tolerance=TOLERANCE)
            assert np.array_equal(policy_vi, policy_pi), (current, discount)
            assert np.allclose(value_vi, value_pi, atol=1e-6, rtol=0), (current, discount)
            exact = np.linalg.solve(
                np.eye(mdp.num_states) - discount * mdp.P[states, policy_vi],
                mdp.R[states, policy_vi],
            )
            # Greediness of the directly evaluated returned policy verifies optimality.
            exact_q = mdp.R + discount * (mdp.P @ exact)
            assert np.array_equal(np.argmax(exact_q, axis=1), policy_vi)
            row = {"current": current, "discount": discount}
            for method, values, policy in (
                ("vi", value_vi, policy_vi),
                ("pi", value_pi, policy_pi),
            ):
                row.update({f"{method}_v{s}": float(v) for s, v in enumerate(values)})
                row[f"{method}_policy"] = " ".join(str(int(a)) for a in policy)
                optimal_backup = np.max(mdp.R + discount * (mdp.P @ values), axis=1)
                residual = float(np.max(np.abs(optimal_backup - values)))
                error = float(np.max(np.abs(exact - values)))
                row[f"{method}_bellman_residual"] = residual
                row[f"{method}_direct_solve_error"] = error
                assert residual <= 1.1 * TOLERANCE + 1e-12
                assert error <= 1.05 * TOLERANCE / (1 - discount) + 1e-9
            row["max_abs_pi_vi_difference"] = float(np.max(np.abs(value_pi - value_vi)))
            rows.append(row)
        print(f"Checked {len(discounts)} discounts for {current} current.", flush=True)
    # Verify the qualitative value ordering separately at every sampled discount.
    for index in range(len(discounts)):
        vals = [rows[c * len(discounts) + index]["vi_v0"] for c in range(3)]
        assert vals[0] + 1e-9 >= vals[1] and vals[1] + 1e-9 >= vals[2]
    return rows


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUTPUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(required: list[dict], extension: list[dict]) -> dict:
    summary = {"tolerance": TOLERANCE, "required_grid_points": len(DISCOUNT_GRID),
               "extension_grid_points": len(EXTENSION_GRID), "currents": {}}
    for current in CURRENTS:
        base = [row for row in required if row["current"] == current]
        fine = [row for row in extension if row["current"] == current]
        left = [row["discount"] for row in base if row["vi_policy"].split()[0] == str(LEFT)]
        right = [row["discount"] for row in base if row["vi_policy"].split()[0] == str(RIGHT)]
        assert max(left) < min(right)
        at99 = next(row for row in base if row["discount"] == 0.99)
        at999 = fine[-1]
        assert at99["vi_policy"] == at999["vi_policy"] == "1 1 1 1 1 1"
        summary["currents"][current] = {
            "largest_left_grid_discount": max(left),
            "first_right_grid_discount": min(right),
            "value_at_0_99": at99["vi_v0"],
            "value_at_0_999": at999["vi_v0"],
            "pi_vi_max_abs_difference_required": max(r["max_abs_pi_vi_difference"] for r in base),
            "pi_vi_max_abs_difference_extension": max(r["max_abs_pi_vi_difference"] for r in fine),
            "extension_left_count": sum(r["vi_policy"].split()[0] == "0" for r in fine),
            "extension_right_count": sum(r["vi_policy"].split()[0] == "1" for r in fine),
        }
    all_rows = required + extension
    summary["total_current_discount_pairs"] = len(all_rows)
    summary["max_abs_pi_vi_difference"] = max(r["max_abs_pi_vi_difference"] for r in all_rows)
    summary["max_bellman_residual"] = max(r[f"{m}_bellman_residual"] for r in all_rows for m in ("pi", "vi"))
    summary["max_direct_solve_error"] = max(r[f"{m}_direct_solve_error"] for r in all_rows for m in ("pi", "vi"))
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def plot_values(extension: list[dict]) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.5), layout="constrained")
    colors = {"weak": "#277da1", "medium": "#e59500", "strong": "#b13c54"}
    for current in CURRENTS:
        rows = [row for row in extension if row["current"] == current]
        discounts = [row["discount"] for row in rows]
        values = [row["vi_v0"] for row in rows]
        for ax in axes:
            ax.plot(discounts, values, color=colors[current], label=current.title(),
                    marker="o", markersize=2.8, linewidth=1.6)
    for ax in axes:
        ax.plot(EXTENSION_GRID, 0.005 / (1 - EXTENSION_GRID), linestyle="--",
                color="#666666", linewidth=1, label="Always LEFT")
        ax.set_xlabel(r"Discount factor $\gamma$")
        ax.set_ylabel(r"Optimal initial value $V^*(s_0)$")
        ax.grid(alpha=0.2)
    axes[0].set_title("Full graduate-extension interval")
    axes[0].set_xlim(0.8, 1.001)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].set_title("Detail: moderate discounts")
    axes[1].set_xlim(0.8, 0.95)
    weak_rows = [row for row in extension if row["current"] == "weak"]
    detail_max = float(np.interp(0.95, EXTENSION_GRID, [row["vi_v0"] for row in weak_rows]))
    axes[1].set_ylim(0, detail_max * 1.13)
    axes[1].axvspan(0.92, 0.93, color=colors["strong"], alpha=0.12)
    axes[1].text(0.925, detail_max * 1.03, "Strong transition\n0.92 to 0.93",
                 ha="center", va="top", color=colors["strong"], fontsize=8)
    fig.savefig(OUTPUT / "values_vs_discount.pdf")
    fig.savefig(OUTPUT / "values_vs_discount.png", dpi=180)
    plt.close(fig)


def scientific(value: float) -> str:
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def write_report(summary: dict) -> None:
    by_current = summary["currents"]
    table_rows = []
    comparison_rows = []
    for current, stats in by_current.items():
        table_rows.append(
            f"{current.title()} & {stats['largest_left_grid_discount']:.2f} & "
            f"{stats['first_right_grid_discount']:.2f} & {stats['value_at_0_99']:.6f} & "
            f"{stats['value_at_0_999']:.6f} \\\\"
        )
        comparison_rows.append(
            f"{current.title()} & ${scientific(stats['pi_vi_max_abs_difference_required'])}$ & "
            f"${scientific(stats['pi_vi_max_abs_difference_extension'])}$ \\\\"
        )
    reduction99 = 100 * (1 - by_current["strong"]["value_at_0_99"] / by_current["weak"]["value_at_0_99"])
    reduction999 = 100 * (1 - by_current["strong"]["value_at_0_999"] / by_current["weak"]["value_at_0_999"])
    tex = r"""\documentclass[11pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{amsmath,booktabs,graphicx,xcolor,hyperref}
\hypersetup{colorlinks=true,urlcolor=blue}
\definecolor{navy}{HTML}{17324D}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\begin{document}
{\LARGE\bfseries\color{navy} Coding Assignment 1}\par
{\large ECE 414/517 --- Dynamic Programming on RiverSwim}\par
\vspace{2pt}\hrule\vspace{6pt}
\textbf{Setup.} The supplied six-state model starts at $s_0$ (array index 0).
Action 0 is LEFT and action 1 is RIGHT. LEFT at $s_0$ pays $0.005$;
RIGHT at $s_5$ pays $1$. Values are expected infinite-horizon discounted
returns. Value iteration and the policy-evaluation step use synchronous
updates, stopping when the maximum absolute change in the value vector is
at most $10^{-10}$. Policy iteration stops when its policy is unchanged.
All supplied iteration limits are retained. Greedy ties select the smallest action index.

\textbf{(a) Largest grid discount selecting LEFT.}
I evaluated every discount in
$\{0.00,0.01,\ldots,0.99\}$ for every current strength.
The initial action is LEFT through the second column of the table and
RIGHT from the third column onward on this grid.
These are \emph{grid thresholds}, not exact continuous switching points.
The corresponding switching intervals are $[0.61,0.62]$, $[0.71,0.72]$,
and $[0.92,0.93]$ for weak, medium, and strong currents.
\begin{center}
\begin{tabular}{lrrrr}
\toprule
Current & Last LEFT $\gamma$ & First RIGHT $\gamma$ & $V^*(s_0)$ at .99 & $V^*(s_0)$ at .999\\
\midrule
@@TABLE@@
\bottomrule
\end{tabular}
\end{center}
The $.999$ values come from the graduate-extension grid. At both $.99$
and $.999$, the optimal policy for all three currents is
$[1,1,1,1,1,1]$: choose RIGHT in every state.

\textbf{(b) Policy iteration versus value iteration.}
Both methods converge for every tested setting and return identical
six-state policy arrays in all 300 required-grid cases and all 150
extension cases. The maximum absolute difference between their value
vectors, over all states and discounts, is shown below.
\begin{center}
\begin{tabular}{lrr}
\toprule
Current & Required grid & Extension grid\\
\midrule
@@COMPARISON@@
\bottomrule
\end{tabular}
\end{center}
The overall difference is $@@DIFFERENCE@@$.
As an independent numerical check, each returned policy was evaluated with
$(I-\gamma P^\pi)^{-1}r^\pi$ using a linear-system solve and confirmed greedy
with respect to those values. Across both methods and both grids, the
largest optimal Bellman residual is $@@RESIDUAL@@$ and the largest difference
from direct evaluation is $@@ERROR@@$.
The stopping tolerance bounds successive updates, not the exact value
error: the contraction bound is approximately
$10^{-10}/(1-\gamma)$, or $10^{-7}$ at $\gamma=.999$.

\textbf{(c) Effect of stronger current.}
The RIGHT dynamics list backward, stay, and forward probabilities:
weak $(.05,.60,.35)$, medium $(.10,.65,.25)$, and strong $(.15,.75,.10)$.
The interior expected displacement under RIGHT
therefore decreases from $+.30$ to $+.15$ to $-.05$ states per step.
Reaching and remaining near the larger right-boundary reward becomes harder,
so a longer effective horizon is needed to justify leaving the small
guaranteed reward on the left. The largest LEFT discount increases by
$0.10$ from weak to medium and by another $0.21$ from medium to strong.
At a fixed sampled discount, $V^*(s_0)$ never increases with current strength:
it is weak $\geq$ medium $\geq$ strong on both grids. At $.99$ the strong-current
value is @@REDUCTION99@@\% below the weak-current value.

\newpage
{\Large\bfseries\color{navy} Graduate extension and reproducibility}\par
\textbf{ECE 517 extension.}
I evaluated 50 evenly spaced discounts using
\texttt{np.linspace(0.80, 0.999, 50)} for each current strength.
This includes both endpoints and exceeds the required 20 points.
The figure plots the initial optimal value from value iteration;
policy iteration gives the same curves to numerical precision.

\begin{center}
\includegraphics[width=\linewidth]{values_vs_discount.pdf}
\end{center}
{\small Each colored marker is one of the 50 tested discounts. The right
panel enlarges the moderate-discount range; the shaded band brackets the
strong-current transition on the required $0.01$ grid. The dashed curve
is the value of staying at $s_0$ and choosing LEFT forever.}\par

For weak and medium currents, all 50 extension discounts choose RIGHT
at $s_0$. For the strong current, @@LEFTCOUNT@@ choose LEFT and
@@RIGHTCOUNT@@ choose RIGHT. Whenever LEFT is optimal at the initial state,
the agent remains there and receives $V^*(s_0)=0.005/(1-\gamma)$.
For example, all three currents have value $0.0125$ at $\gamma=.60$;
the stronger currents separate only when swimming right becomes worthwhile.

As $\gamma$ approaches 1, future rewards retain more weight and the
effective horizon $1/(1-\gamma)$ grows from 5 at $.80$ to 1,000 at $.999$.
Repeated rewards at the right boundary then compensate for travel delays,
even against the strong current. All three values rise sharply, while the
weak current still yields the highest return because it reaches and occupies
the right boundary more easily. The strong-current value remains
@@REDUCTION999@@\% below the weak-current value at $.999$.
The LEFT baseline also diverges as $\gamma$ approaches 1, so rising value
alone does not imply a policy change; the relative action values determine it.
\end{document}
"""
    replacements = {
        "@@TABLE@@": "\n".join(table_rows),
        "@@COMPARISON@@": "\n".join(comparison_rows),
        "@@DIFFERENCE@@": scientific(summary["max_abs_pi_vi_difference"]),
        "@@RESIDUAL@@": scientific(summary["max_bellman_residual"]),
        "@@ERROR@@": scientific(summary["max_direct_solve_error"]),
        "@@REDUCTION99@@": f"{reduction99:.1f}",
        "@@REDUCTION999@@": f"{reduction999:.1f}",
        "@@LEFTCOUNT@@": str(by_current["strong"]["extension_left_count"]),
        "@@RIGHTCOUNT@@": str(by_current["strong"]["extension_right_count"]),
    }
    for placeholder, value in replacements.items():
        tex = tex.replace(placeholder, value)
    source = OUTPUT / "coding_assignment1_report.tex"
    source.write_text(tex, encoding="utf-8")
    if shutil.which("pdflatex") is None:
        raise RuntimeError("Data, plot and TeX were generated; install pdflatex to build the report PDF.")
    with tempfile.TemporaryDirectory(prefix="riverswim_report_") as build:
        command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                   "-output-directory", build, str(source)]
        result = subprocess.run(command, cwd=OUTPUT, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        shutil.copy2(Path(build) / "coding_assignment1_report.pdf", OUTPUT)
        # Surface typesetting issues for inspection without retaining auxiliary files.
        for line in result.stdout.splitlines():
            if "Overfull" in line or "Underfull" in line:
                print(line)


def main() -> None:
    required = evaluate_grid(DISCOUNT_GRID)
    extension = evaluate_grid(EXTENSION_GRID)
    write_csv("required_grid.csv", required)
    write_csv("extension_grid.csv", extension)
    summary = summarize(required, extension)
    plot_values(extension)
    write_report(summary)
    print(json.dumps(summary, indent=2))
    print(f"Report: {OUTPUT / 'coding_assignment1_report.pdf'}")


if __name__ == "__main__":
    main()

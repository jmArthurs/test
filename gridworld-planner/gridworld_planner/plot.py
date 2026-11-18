"""Plot success rate vs. task length, one line per (planner, feedback) condition.

    python -m gridworld_planner.plot results/results.csv results/success_vs_length.png

Error bars are 95% Wilson score intervals for a binomial proportion (well behaved at 0% and 100%).
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

# Categorical slots in fixed order (validated palette); text and grid stay neutral.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PLANNER_ORDER = ["oracle", "random", "worldmodel"]  # LLM planners follow, in order of first appearance


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(csv_path: str | Path) -> dict:
    """{(planner, feedback): {optimal_steps: (k_success, n)}}"""
    agg: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            cell = agg[(row["planner"], row["feedback"] == "True")][int(row["optimal_steps"])]
            cell[0] += row["success"] == "True"
            cell[1] += 1
    return {cond: {x: tuple(v) for x, v in sorted(d.items())} for cond, d in agg.items()}


def plot_results(csv_path: str | Path, png_path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = summarize(csv_path)
    planners = [p for p in PLANNER_ORDER if any(c[0] == p for c in data)]
    planners += sorted({c[0] for c in data} - set(planners))
    color = {p: SERIES[i % len(SERIES)] for i, p in enumerate(planners)}
    conditions = [(p, fb) for p in planners for fb in (True, False) if (p, fb) in data]
    xs = sorted({x for d in data.values() for x in d})
    ns = sorted({n for d in data.values() for _, n in d.values()})

    plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT_2,
                         "xtick.color": TEXT_2, "ytick.color": TEXT_2})
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    width = 0.12 * (len(conditions) - 1)  # dodge so error bars don't overlap
    for i, (planner, fb) in enumerate(conditions):
        dx = -width / 2 + i * 0.12 if len(conditions) > 1 else 0.0
        pts = data[(planner, fb)]
        x = [s + dx for s in pts]
        rate = [k / n for k, n in pts.values()]
        lo_hi = [wilson(k, n) for k, n in pts.values()]
        err = [[r - lo for r, (lo, _) in zip(rate, lo_hi)], [hi - r for r, (_, hi) in zip(rate, lo_hi)]]
        ax.errorbar(x, rate, yerr=err, color=color[planner], lw=2, capsize=3, elinewidth=1.2,
                    ls="-" if fb else "--", marker="o", ms=7, mew=1.5,
                    mfc=color[planner] if fb else SURFACE, mec=color[planner],
                    label=f"{planner} · feedback {'on' if fb else 'off'}")
    ax.set_xticks(xs)
    ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7)
    ax.set_ylim(-0.04, 1.06)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Task length (optimal number of steps)")
    ax.set_ylabel("Success rate")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    n_text = f"n = {ns[0]}" if len(ns) == 1 else f"n = {ns[0]}–{ns[-1]}"
    ax.set_title("Planner success rate vs. task length", loc="left", fontsize=13, color=TEXT, pad=22)
    ax.text(0, 1.02, f"{n_text} episodes per point · bars: 95% Wilson interval · "
            "solid/filled = feedback on, dashed/hollow = off",
            transform=ax.transAxes, fontsize=8.5, color=TEXT_2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
    fig.tight_layout()
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m gridworld_planner.plot RESULTS.csv OUT.png")
    plot_results(sys.argv[1], sys.argv[2])

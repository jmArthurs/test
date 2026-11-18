"""Run N episodes per task for each planner, with and without failure feedback, and log to CSV.

    python -m gridworld_planner.evaluate --episodes 20                      # random, oracle, worldmodel
    python -m gridworld_planner.evaluate --planners llm --episodes 5        # LLM (costs money on api)
    python -m gridworld_planner.evaluate --planners llm --backend local     # LLM via Ollama

The LLM planner only runs when listed in --planners. Rows are written as each episode
finishes; --resume skips (task, planner, feedback, episode) rows already in the CSV, so an
interrupted LLM run can be continued.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import fields
from pathlib import Path

from .llm import load_config, make_backend
from .planners import LLMPlanner, OraclePlanner, RandomPlanner, WorldModelPlanner
from .runner import EpisodeResult, run_episode
from .tasks import TASKS, TASKS_BY_NAME

COLUMNS = [f.name for f in fields(EpisodeResult)]
FEEDBACK = {"both": (False, True), "on": (True,), "off": (False,)}


def make_planners(names: list[str], config_path: str | None, backend: str | None):
    out = []
    for name in names:
        if name == "random":
            out.append(RandomPlanner())
        elif name == "oracle":
            out.append(OraclePlanner())
        elif name == "worldmodel":
            out.append(WorldModelPlanner())
        elif name == "llm":
            cfg = load_config(config_path, backend)
            planner = LLMPlanner(make_backend(cfg))
            planner.name = f"llm:{cfg.backend}:{cfg.model}"
            out.append(planner)
        else:
            raise SystemExit(f"unknown planner {name!r}; choose from random, oracle, worldmodel, llm")
    return out


def _key(row: dict) -> tuple:
    return row["task"], row["planner"], str(row["feedback"]), str(row["episode"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=10, help="episodes per task per condition")
    ap.add_argument("--planners", default="random,oracle,worldmodel",
                    help="comma list of random,oracle,worldmodel,llm")
    ap.add_argument("--feedback", choices=FEEDBACK, default="both")
    ap.add_argument("--tasks", default=",".join(t.name for t in TASKS), help="comma list of task names")
    ap.add_argument("--backend", choices=("api", "local"), help="override [llm] backend in the config")
    ap.add_argument("--config", help="TOML config path (default: config.toml in the project root)")
    ap.add_argument("--out", default="results/results.csv")
    ap.add_argument("--resume", action="store_true", help="append, skipping episodes already logged")
    ap.add_argument("--plot", default="results/success_vs_length.png", help="PNG path; '' to skip")
    args = ap.parse_args(argv)

    tasks = [TASKS_BY_NAME[n] for n in args.tasks.split(",")]
    planners = make_planners(args.planners.split(","), args.config, args.backend)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out.exists():
        with out.open() as f:
            done = {_key(r) for r in csv.DictReader(f)}
    write_header = not (args.resume and out.exists())
    with out.open("a" if args.resume else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        for planner in planners:
            for feedback in FEEDBACK[args.feedback]:
                for task in tasks:
                    for ep in range(args.episodes):
                        if (task.name, planner.name, str(feedback), str(ep)) in done:
                            continue
                        r = run_episode(task, planner, feedback, ep)
                        writer.writerow(r.row())
                        f.flush()
                        print(f"{planner.name:<28} feedback={'on ' if feedback else 'off'} "
                              f"{task.name:<16} ep={ep:<3} success={r.success!s:<5} "
                              f"steps={r.steps_used}/{r.max_steps}", file=sys.stderr)
    print(f"wrote {out}", file=sys.stderr)
    if args.plot:
        from .plot import plot_results
        plot_results(out, args.plot)
        print(f"wrote {args.plot}", file=sys.stderr)


if __name__ == "__main__":
    main()

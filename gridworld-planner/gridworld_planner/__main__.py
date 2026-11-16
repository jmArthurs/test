"""Watch the oracle solve a map, or play it yourself.

    python -m gridworld_planner                          # default demo map
    python -m gridworld_planner --task two_room_chain    # one of the evaluation tasks
    python -m gridworld_planner --task walk_to_door --play
"""
import argparse

from . import ACTIONS, GridWorld, plan
from .tasks import TASKS_BY_NAME


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=TASKS_BY_NAME, help="run an evaluation task instead of the demo map")
    ap.add_argument("--play", action="store_true", help="type the actions yourself")
    args = ap.parse_args()
    task = TASKS_BY_NAME.get(args.task)
    env = task.make_env() if task else GridWorld()
    success = task.success if task else (lambda e: e.reached_goal)
    if task:
        print(f"Task: {task.description} (optimal {task.optimal_steps} steps, limit {task.max_steps})")
    print(env.render(), "\n" + env.observe(), sep="\n")
    if args.play:
        while not env.done and not success(env):
            action = input(f"\naction {ACTIONS}> ").strip()
            print(env.step(action).observation if action in ACTIONS else "Unknown action.")
    else:
        actions = plan(env, success)
        print(f"\nPlan ({len(actions)} steps): {' '.join(actions)}")
        for action in actions:
            print(f"\n> {action}\n{env.step(action).observation}")
    print("\nSuccess!" if success(env) else "\nFailed.")


if __name__ == "__main__":
    main()

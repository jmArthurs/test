"""Run `python -m gridworld_planner` to watch the planner, or add --play to play yourself."""
import sys

from . import ACTIONS, GridWorld, plan


def main() -> None:
    env = GridWorld()
    print(env.render(), "\n" + env.observe(), sep="\n")
    if "--play" in sys.argv:
        while not env.done:
            action = input(f"\naction {ACTIONS}> ").strip()
            if action in ACTIONS:
                print(env.step(action).observation)
            else:
                print("Unknown action.")
        return
    actions = plan(env)
    print(f"\nPlan ({len(actions)} steps): {' '.join(actions)}")
    for action in actions:
        print(f"\n> {action}\n{env.step(action).observation}")


if __name__ == "__main__":
    main()

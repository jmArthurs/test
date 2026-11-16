"""A ladder of fixed tasks whose optimal solutions take 1, 3, 5, 7 and 10 steps."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .env import GridWorld


def facing_door(env: GridWorld) -> bool:
    return env.cell(env._ahead()) in "Dd"


def at_door_with_key(env: GridWorld) -> bool:
    return "key" in env.holding and facing_door(env)


def reached_goal(env: GridWorld) -> bool:
    return env.reached_goal


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    layout: str
    optimal_steps: int
    max_steps: int
    seed: int
    success: Callable[[GridWorld], bool]
    start: tuple[int, int] | None = None

    def make_env(self) -> GridWorld:
        return GridWorld(self.layout, max_steps=self.max_steps, seed=self.seed, start=self.start)


TASKS = [
    Task("walk_to_door", "Walk up to the door and face it.",
         "#####\n#@.D#\n#####", 1, 5, 101, facing_door),
    Task("key_then_door", "Pick up the key, then walk up to the door.",
         "######\n#@k.D#\n######", 3, 10, 102, at_door_with_key),
    Task("unlock_and_pass", "Pick up the key, unlock the door and go through to the goal.",
         "######\n#@kDG#\n######", 5, 15, 103, reached_goal),
    Task("switch_key_door", "Press the switch you start on, then key, door, gate, goal.",
         "#######\n#skDXG#\n#######", 7, 20, 104, reached_goal, start=(1, 1)),
    Task("two_room_chain", "Room 1 has the switch and key; the door leads to room 2, "
         "whose gate guards the goal.",
         "#########\n#.s@#####\n#.k.D.XG#\n#########", 10, 30, 105, reached_goal),
]
TASKS_BY_NAME = {t.name: t for t in TASKS}

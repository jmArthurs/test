"""Breadth-first planner that finds a shortest action sequence to the goal."""
from __future__ import annotations

import copy
from collections import deque
from typing import Callable

from .env import ACTIONS, GridWorld


def plan(env: GridWorld, success: Callable[[GridWorld], bool] | None = None,
         max_nodes: int = 100_000) -> list[str] | None:
    """Return the shortest list of actions from env's current state until ``success``
    holds (default: the goal is reached), or None if no plan exists within the step limit.

    The given env is not modified; search runs on copies.
    """
    success = success or (lambda e: e.reached_goal)
    start = copy.deepcopy(env)
    if success(start):
        return []
    frontier = deque([(start, [])])
    seen = {start.state_key()}
    while frontier and len(seen) <= max_nodes:
        node, path = frontier.popleft()
        for action in ACTIONS:
            child = copy.deepcopy(node)
            child.step(action)
            if success(child):
                return path + [action]
            key = child.state_key()
            if child.done or key in seen:
                continue
            seen.add(key)
            frontier.append((child, path + [action]))
    return None

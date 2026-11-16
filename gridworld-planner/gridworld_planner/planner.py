"""Breadth-first planner that finds a shortest action sequence to the goal."""
from __future__ import annotations

import copy
from collections import deque

from .env import ACTIONS, GridWorld


def plan(env: GridWorld, max_nodes: int = 100_000) -> list[str] | None:
    """Return the shortest list of actions from env's current state to the goal, or None.

    The given env is not modified; search runs on copies.
    """
    start = copy.deepcopy(env)
    if start.done:
        return [] if start.reached_goal else None
    frontier = deque([(start, [])])
    seen = {start.state_key()}
    while frontier and len(seen) <= max_nodes:
        node, path = frontier.popleft()
        for action in ACTIONS:
            child = copy.deepcopy(node)
            child.step(action)
            if child.reached_goal:
                return path + [action]
            key = child.state_key()
            if child.done or key in seen:
                continue
            seen.add(key)
            frontier.append((child, path + [action]))
    return None

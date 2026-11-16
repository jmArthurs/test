"""Run one episode of a task with a planner, optionally feeding back failures."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .env import ACTIONS
from .planners import Planner
from .tasks import Task


@dataclass
class EpisodeResult:
    task: str
    optimal_steps: int
    planner: str
    feedback: bool
    episode: int
    seed: int
    success: bool
    steps_used: int
    max_steps: int
    failed_actions: int   # actions that had no effect (bumps, locked door, nothing to use...)
    invalid_replies: int  # planner outputs that were not an action (LLM only)
    model: str

    def row(self) -> dict:
        return asdict(self)


def episode_seed(task: Task, episode: int) -> int:
    """Fixed, distinct seed per (task, episode); identical across planners and conditions."""
    return task.seed * 1000 + episode


def run_episode(task: Task, planner: Planner, feedback: bool, episode: int = 0) -> EpisodeResult:
    seed = episode_seed(task, episode)
    env = task.make_env(seed)
    planner.reset(task, env, seed)
    note, failed, invalid = None, 0, 0
    while not env.done and not task.success(env):
        action = planner.act(env.observe(), note if feedback else None)
        if action in ACTIONS:
            res = env.step(action)
            note = f'Your last action "{action}" failed: {res.message}' if res.failed else None
        else:
            invalid += 1
            res = env.waste_step("invalid reply")
            note = ("Your last reply did not end with a valid action line, so the step was wasted. "
                    "End with e.g. ACTION: up")
        failed += res.failed and action in ACTIONS
    return EpisodeResult(task.name, task.optimal_steps, planner.name, feedback, episode, seed,
                         task.success(env), env.steps, task.max_steps, failed, invalid, planner.model)
